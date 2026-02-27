'use client';
// frontend/app/components/MapView.tsx

import L, {
  type LatLngBoundsExpression,
  type LatLngExpression,
  type LeafletMouseEvent,
} from 'leaflet';
import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from 'react';
import {
  Circle,
  CircleMarker,
  MapContainer,
  Marker,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from 'react-leaflet';

import {
  buildIncidentOverlayPoints,
  buildPreviewRouteNodes,
  buildPreviewRouteSegmentsFromNodes,
  buildSegmentBuckets,
  buildStopOverlayPoints,
} from '../lib/mapOverlays';
import type {
  DutyChainStop,
  IncidentEventType,
  LatLng,
  MapFailureOverlay,
  ManagedStop,
  PinFocusRequest,
  PinSelectionId,
  RouteOption,
  TutorialGuideTarget,
} from '../lib/types';

export type MarkerKind = 'origin' | 'destination';
const MAX_POLYLINE_POINTS = 1000;

type Props = {
  origin: LatLng | null;
  destination: LatLng | null;
  tutorialDraftOrigin?: LatLng | null;
  tutorialDraftDestination?: LatLng | null;
  tutorialDragDraftOrigin?: LatLng | null;
  tutorialDragDraftDestination?: LatLng | null;
  managedStop?: ManagedStop | null;
  originLabel?: string;
  destinationLabel?: string;

  selectedPinId?: PinSelectionId | null;
  focusPinRequest?: PinFocusRequest | null;
  fitAllRequestNonce?: number;

  route: RouteOption | null;
  failureOverlay?: MapFailureOverlay | null;
  timeLapsePosition?: LatLng | null;
  dutyStops?: DutyChainStop[];
  showStopOverlay?: boolean;
  showIncidentOverlay?: boolean;
  showSegmentTooltips?: boolean;
  showPreviewConnector?: boolean;
  overlayLabels?: {
    stopLabel: string;
    segmentLabel: string;
    incidentTypeLabels: Record<'dwell' | 'accident' | 'closure', string>;
  };
  onTutorialAction?: (actionId: string) => void;
  onTutorialTargetState?: (state: { hasSegmentTooltipPath: boolean; hasIncidentMarkers: boolean }) => void;
  tutorialMapLocked?: boolean;
  tutorialMapDimmed?: boolean;
  tutorialViewportLocked?: boolean;
  tutorialHideZoomControls?: boolean;
  tutorialRelaxBounds?: boolean;
  tutorialExpectedAction?: string | null;
  tutorialGuideTarget?: TutorialGuideTarget | null;
  tutorialGuideVisible?: boolean;
  tutorialConfirmPin?: 'origin' | 'destination' | null;
  tutorialDragConfirmPin?: 'origin' | 'destination' | null;

  onMapClick: (lat: number, lon: number) => void;
  onSelectPinId?: (id: PinSelectionId | null) => void;
  onMoveMarker: (kind: MarkerKind, lat: number, lon: number) => boolean;
  onMoveStop?: (lat: number, lon: number) => boolean;
  onAddStopFromPin?: () => void;
  onRenameStop?: (name: string) => void;
  onDeleteStop?: () => void;
  onFocusPin?: (id: PinSelectionId) => void;
  onSwapMarkers?: () => void;
  onTutorialConfirmPin?: (kind: 'origin' | 'destination') => void;
  onTutorialConfirmDrag?: (kind: 'origin' | 'destination') => void;
};

// Rough UK bounds (keeps the demo focused on UK routing).
const UK_BOUNDS: LatLngBoundsExpression = [
  [49.5, -8.7],
  [61.1, 2.1],
];
const WORLD_BOUNDS: LatLngBoundsExpression = [
  [-85, -180],
  [85, 180],
];

// Fix for Leaflet + Next.js Fast Refresh:
// When the module hot-reloads, Leaflet can try to re-use the same DOM container and throws:
//   "Map container is being reused by another instance"
// Forcing the MapContainer to remount on hot reload avoids that.
const MAP_HOT_RELOAD_KEY =
  typeof window === 'undefined'
    ? 'ssr'
    : (() => {
        const w = window as any;
        w.__LEAFLET_MAP_KEY__ = (w.__LEAFLET_MAP_KEY__ ?? 0) + 1;
        return String(w.__LEAFLET_MAP_KEY__);
      })();

function makePinIcon(kind: MarkerKind, selected: boolean) {
  const label = kind === 'origin' ? 'Start' : 'End';
  const assistive = kind === 'origin' ? 'Start marker' : 'End marker';
  const baseClass = kind === 'origin' ? 'pin-origin' : 'pin-destination';
  const selectedClass = selected ? 'pin--selected' : '';

  return L.divIcon({
    className: `pin ${baseClass} ${selectedClass}`.trim(),
    html: [
      '<span class="pin__halo" aria-hidden="true"></span>',
      '<span class="pin__body">',
      `<span class="pin__inner">${label}</span>`,
      '</span>',
      '<span class="pin__tail" aria-hidden="true"></span>',
      `<span class="pin__sr">${assistive}</span>`,
    ].join(''),
    iconSize: [42, 58],
    iconAnchor: [21, 56],
    popupAnchor: [0, -48],
  });
}

function ClickHandler({ onMapClick }: { onMapClick: (lat: number, lon: number) => void }) {
  const map = useMap();
  useMapEvents({
    click(e: LeafletMouseEvent) {
      map.stop();
      onMapClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

function FocusPinRequestHandler({
  request,
  origin,
  destination,
  managedStop,
  originRef,
  destRef,
  stopRef,
}: {
  request: PinFocusRequest | null;
  origin: LatLng | null;
  destination: LatLng | null;
  managedStop: ManagedStop | null;
  originRef: RefObject<L.Marker | null>;
  destRef: RefObject<L.Marker | null>;
  stopRef: RefObject<L.Marker | null>;
}) {
  const map = useMap();

  useEffect(() => {
    if (!request) return;
    let target: LatLng | null = null;
    if (request.id === 'origin') target = origin;
    else if (request.id === 'destination') target = destination;
    else target = managedStop ? { lat: managedStop.lat, lon: managedStop.lon } : null;
    if (!target) return;

    const nextZoom =
      request.zoom !== undefined
        ? request.zoom
        : Math.min(14, Math.max(map.getZoom() + 1, 11));
    map.flyTo([target.lat, target.lon], nextZoom, {
      animate: true,
      duration: 0.45,
    });

    if (request.openPopup === false) {
      return;
    }

    const timer = window.setTimeout(() => {
      try {
        if (request.id === 'origin') {
          originRef.current?.openPopup();
        } else if (request.id === 'destination') {
          destRef.current?.openPopup();
        } else {
          stopRef.current?.openPopup();
        }
      } catch {
        // no-op
      }
    }, 140);

    return () => window.clearTimeout(timer);
  }, [
    request?.nonce,
    request?.id,
    map,
    origin,
    destination,
    managedStop,
    originRef,
    destRef,
    stopRef,
  ]);

  return null;
}

function FitAllRequestHandler({
  nonce,
  origin,
  destination,
  managedStop,
  route,
  lockViewport,
}: {
  nonce: number;
  origin: LatLng | null;
  destination: LatLng | null;
  managedStop: ManagedStop | null;
  route: RouteOption | null;
  lockViewport: boolean;
}) {
  const map = useMap();
  const lastHandledNonceRef = useRef(nonce);
  const logFitDebug = useCallback(
    (_event: string, _payload?: Record<string, unknown>) => {},
    [],
  );

  useEffect(() => {
    logFitDebug('effect:enter', {
      nonce,
      lastHandledNonce: lastHandledNonceRef.current,
      lockViewport,
      hasOrigin: Boolean(origin),
      hasDestination: Boolean(destination),
      hasStop: Boolean(managedStop),
      routeId: route?.id ?? null,
      routeGeomPoints: route?.geometry?.coordinates?.length ?? 0,
      zoomBefore: map.getZoom(),
      centerBefore: map.getCenter(),
    });
    if (nonce <= 0) {
      logFitDebug('effect:skip:nonce<=0');
      return;
    }
    if (nonce === lastHandledNonceRef.current) {
      logFitDebug('effect:skip:nonce-unchanged');
      return;
    }
    lastHandledNonceRef.current = nonce;
    const points: LatLngExpression[] = [];

    if (origin) points.push([origin.lat, origin.lon]);
    if (managedStop) points.push([managedStop.lat, managedStop.lon]);
    if (destination) points.push([destination.lat, destination.lon]);

    const coords = route?.geometry?.coordinates ?? [];
    if (Array.isArray(coords) && coords.length > 1) {
      const step = Math.max(1, Math.floor(coords.length / 50));
      for (let idx = 0; idx < coords.length; idx += step) {
        const [lon, lat] = coords[idx];
        points.push([lat, lon]);
      }
      const [lastLon, lastLat] = coords[coords.length - 1];
      points.push([lastLat, lastLon]);
    }
    logFitDebug('effect:points-built', {
      pointsCount: points.length,
      origin,
      destination,
      managedStop,
      sampledRoutePointsCount: coords.length > 1 ? points.length - (origin ? 1 : 0) - (managedStop ? 1 : 0) - (destination ? 1 : 0) : 0,
      firstPoint: points[0] ?? null,
      lastPoint: points[points.length - 1] ?? null,
    });

    const getFitPadding = () => {
      const basePadX = lockViewport ? 26 : 36;
      const basePadY = lockViewport ? 26 : 36;
      const paddingTopLeft: [number, number] = [basePadX, basePadY];
      const paddingBottomRight: [number, number] = [basePadX, basePadY];
      const mapRect = map.getContainer().getBoundingClientRect();
      const maxPadX = lockViewport
        ? Math.max(260, Math.min(560, Math.floor(mapRect.width * 0.48)))
        : Math.max(36, Math.floor(mapRect.width * 0.33));
      const maxPadY = lockViewport
        ? Math.max(90, Math.min(220, Math.floor(mapRect.height * 0.28)))
        : Math.max(36, Math.floor(mapRect.height * 0.28));
      const occluders: HTMLElement[] = [];
      const occluderDebug: Array<Record<string, unknown>> = [];
      const panel = document.querySelector<HTMLElement>('.panel:not(.isCollapsed)');
      if (panel) occluders.push(panel);
      const tutorialCard = document.querySelector<HTMLElement>(
        '.tutorialOverlay.isRunning .tutorialOverlay__card',
      );
      if (tutorialCard) occluders.push(tutorialCard);
      logFitDebug('padding:occluder-scan', {
        mapRect: {
          left: mapRect.left,
          top: mapRect.top,
          width: mapRect.width,
          height: mapRect.height,
          right: mapRect.right,
          bottom: mapRect.bottom,
        },
        maxPadX,
        maxPadY,
        lockViewport,
        occluderCount: occluders.length,
        occluderClasses: occluders.map((node) => node.className),
      });

      if (lockViewport) {
        for (const occluder of occluders) {
          const rect = occluder.getBoundingClientRect();
          const overlapLeft = Math.max(mapRect.left, rect.left);
          const overlapRight = Math.min(mapRect.right, rect.right);
          const overlapTop = Math.max(mapRect.top, rect.top);
          const overlapBottom = Math.min(mapRect.bottom, rect.bottom);
          const overlapWidth = overlapRight - overlapLeft;
          const overlapHeight = overlapBottom - overlapTop;
          if (overlapWidth <= 0 || overlapHeight <= 0) {
            occluderDebug.push({
              className: occluder.className,
              mode: 'lockViewport-side-only',
              skipped: 'no-overlap',
              overlapWidth,
              overlapHeight,
            });
            continue;
          }

          const centerX = overlapLeft + overlapWidth / 2;
          const onLeft = centerX < mapRect.left + mapRect.width / 2;
          const candidatePadX = Math.ceil(
            onLeft
              ? overlapRight - mapRect.left + 20
              : mapRect.right - overlapLeft + 20,
          );
          occluderDebug.push({
            className: occluder.className,
            mode: 'lockViewport-side-only',
            overlap: {
              left: overlapLeft,
              top: overlapTop,
              right: overlapRight,
              bottom: overlapBottom,
              width: overlapWidth,
              height: overlapHeight,
            },
            onLeft,
            candidatePadX,
          });
          if (onLeft) {
            paddingTopLeft[0] = Math.max(
              paddingTopLeft[0],
              Math.min(candidatePadX, maxPadX),
            );
          } else {
            paddingBottomRight[0] = Math.max(
              paddingBottomRight[0],
              Math.min(candidatePadX, maxPadX),
            );
          }
        }

        const lane = {
          left: paddingTopLeft[0],
          right: Math.max(paddingTopLeft[0], mapRect.width - paddingBottomRight[0]),
          top: paddingTopLeft[1],
          bottom: Math.max(paddingTopLeft[1], mapRect.height - paddingBottomRight[1]),
        };
        const laneCenter = {
          x: (lane.left + lane.right) / 2,
          y: (lane.top + lane.bottom) / 2,
        };
        const clampToRange = (value: number, min: number, max: number) =>
          Math.max(min, Math.min(max, value));
        const tutorialMidpoint = tutorialCard
          ? {
              x: clampToRange(
                tutorialCard.getBoundingClientRect().right - mapRect.left,
                0,
                mapRect.width,
              ),
              y: clampToRange(
                (tutorialCard.getBoundingClientRect().top +
                  tutorialCard.getBoundingClientRect().bottom) /
                  2 -
                  mapRect.top,
                0,
                mapRect.height,
              ),
            }
          : null;
        const sidebarMidpoint = panel
          ? {
              x: clampToRange(
                panel.getBoundingClientRect().left - mapRect.left,
                0,
                mapRect.width,
              ),
              y: clampToRange(
                (panel.getBoundingClientRect().top + panel.getBoundingClientRect().bottom) / 2 -
                  mapRect.top,
                0,
                mapRect.height,
              ),
            }
          : null;
        const hasUsableCorridor =
          Boolean(tutorialMidpoint && sidebarMidpoint) &&
          (sidebarMidpoint?.x ?? 0) - (tutorialMidpoint?.x ?? 0) > 24;
        const targetCenter = hasUsableCorridor
          ? {
              x: ((tutorialMidpoint?.x ?? laneCenter.x) + (sidebarMidpoint?.x ?? laneCenter.x)) / 2,
              y: ((tutorialMidpoint?.y ?? laneCenter.y) + (sidebarMidpoint?.y ?? laneCenter.y)) / 2,
            }
          : laneCenter;
        return {
          paddingTopLeft,
          paddingBottomRight,
          debug: {
            mode: 'lockViewport-side-only',
            mapRect: {
              width: mapRect.width,
              height: mapRect.height,
            },
            occluders: occluderDebug,
            lane,
            laneCenter,
            targetCenter,
            corridorAnchors: {
              tutorialMidpoint,
              sidebarMidpoint,
              hasUsableCorridor,
              tutorialRightX: tutorialMidpoint?.x ?? null,
              sidebarLeftX: sidebarMidpoint?.x ?? null,
              corridorWidth:
                tutorialMidpoint && sidebarMidpoint
                  ? sidebarMidpoint.x - tutorialMidpoint.x
                  : null,
              targetCenterX: targetCenter.x,
              targetCenterY: targetCenter.y,
            },
          },
        };
      }

      for (const occluder of occluders) {
        const rect = occluder.getBoundingClientRect();
        const overlapLeft = Math.max(mapRect.left, rect.left);
        const overlapRight = Math.min(mapRect.right, rect.right);
        const overlapTop = Math.max(mapRect.top, rect.top);
        const overlapBottom = Math.min(mapRect.bottom, rect.bottom);
        const overlapWidth = overlapRight - overlapLeft;
        const overlapHeight = overlapBottom - overlapTop;
        if (overlapWidth <= 0 || overlapHeight <= 0) {
          occluderDebug.push({
            className: occluder.className,
            skipped: 'no-overlap',
            overlapWidth,
            overlapHeight,
          });
          continue;
        }

        const centerX = overlapLeft + overlapWidth / 2;
        const centerY = overlapTop + overlapHeight / 2;
        const onLeft = centerX < mapRect.left + mapRect.width / 2;
        const onTop = centerY < mapRect.top + mapRect.height / 2;
        const occluderPadX = Math.min(
          Math.ceil(overlapWidth * 0.72 + 16),
          maxPadX,
        );
        const occluderPadY = Math.min(
          Math.ceil(overlapHeight * 0.72 + 14),
          maxPadY,
        );
        occluderDebug.push({
          className: occluder.className,
          overlap: {
            left: overlapLeft,
            top: overlapTop,
            width: overlapWidth,
            height: overlapHeight,
            right: overlapRight,
            bottom: overlapBottom,
          },
          onLeft,
          onTop,
          occluderPadX,
          occluderPadY,
        });

        if (onLeft) {
          paddingTopLeft[0] = Math.max(
            paddingTopLeft[0],
            occluderPadX,
          );
        } else {
          paddingBottomRight[0] = Math.max(
            paddingBottomRight[0],
            occluderPadX,
          );
        }
        if (onTop) {
          paddingTopLeft[1] = Math.max(
            paddingTopLeft[1],
            occluderPadY,
          );
        } else {
          paddingBottomRight[1] = Math.max(
            paddingBottomRight[1],
            occluderPadY,
          );
        }
      }

      const lane = {
        left: paddingTopLeft[0],
        right: Math.max(paddingTopLeft[0], mapRect.width - paddingBottomRight[0]),
        top: paddingTopLeft[1],
        bottom: Math.max(paddingTopLeft[1], mapRect.height - paddingBottomRight[1]),
      };
      const laneCenter = {
        x: (lane.left + lane.right) / 2,
        y: (lane.top + lane.bottom) / 2,
      };
      return {
        paddingTopLeft,
        paddingBottomRight,
        debug: {
          mode: 'default-occluder',
          mapRect: {
            width: mapRect.width,
            height: mapRect.height,
          },
          occluders: occluderDebug,
          lane,
          laneCenter,
        },
      };
    };

    if (points.length >= 2) {
      const sourceBounds = L.latLngBounds(points as LatLngExpression[]);
      const pathLatLngs = (points as [number, number][]).map(([lat, lon]) => L.latLng(lat, lon));
      const routeMidpointLatLng = (() => {
        if (!pathLatLngs.length) return sourceBounds.getCenter();
        if (pathLatLngs.length === 1) return pathLatLngs[0];
        const segmentLengths: number[] = [];
        let totalLength = 0;
        for (let i = 1; i < pathLatLngs.length; i += 1) {
          const len = pathLatLngs[i - 1].distanceTo(pathLatLngs[i]);
          segmentLengths.push(len);
          totalLength += len;
        }
        if (totalLength <= 0) return sourceBounds.getCenter();
        let remaining = totalLength / 2;
        for (let i = 0; i < segmentLengths.length; i += 1) {
          const segLen = segmentLengths[i];
          if (remaining <= segLen) {
            const a = pathLatLngs[i];
            const b = pathLatLngs[i + 1];
            const ratio = segLen <= 0 ? 0 : remaining / segLen;
            return L.latLng(
              a.lat + (b.lat - a.lat) * ratio,
              a.lng + (b.lng - a.lng) * ratio,
            );
          }
          remaining -= segLen;
        }
        return pathLatLngs[pathLatLngs.length - 1];
      })();
      const { paddingTopLeft, paddingBottomRight, debug } = getFitPadding();
      logFitDebug('fitBounds:before', {
        paddingTopLeft,
        paddingBottomRight,
        sourceBounds: {
          south: sourceBounds.getSouth(),
          west: sourceBounds.getWest(),
          north: sourceBounds.getNorth(),
          east: sourceBounds.getEast(),
          center: sourceBounds.getCenter(),
        },
        debug,
      });
      map.once('moveend', () => {
        const nw = map.latLngToContainerPoint(sourceBounds.getNorthWest());
        const se = map.latLngToContainerPoint(sourceBounds.getSouthEast());
        const minX = Math.min(nw.x, se.x);
        const maxX = Math.max(nw.x, se.x);
        const minY = Math.min(nw.y, se.y);
        const maxY = Math.max(nw.y, se.y);
        const lane = debug.lane;
        const laneCenter = debug.laneCenter;
        const targetCenter =
          'targetCenter' in debug && debug.targetCenter ? debug.targetCenter : laneCenter;
        const sourceCenterPx = map.latLngToContainerPoint(sourceBounds.getCenter());
        const routeMidpointPx = map.latLngToContainerPoint(routeMidpointLatLng);
        const centerDelta = {
          dx: routeMidpointPx.x - targetCenter.x,
          dy: routeMidpointPx.y - targetCenter.y,
        };
        logFitDebug('fitBounds:after-moveend', {
          zoomAfter: map.getZoom(),
          centerAfter: map.getCenter(),
          sourcePixelBounds: { minX, minY, maxX, maxY },
          sourceCenterPx,
          routeMidpointLatLng,
          routeMidpointPx,
          lane,
          laneCenter,
          targetCenter,
          centerDeltaPx: centerDelta,
          outsideLane: {
            left: minX < lane.left,
            right: maxX > lane.right,
            top: minY < lane.top,
            bottom: maxY > lane.bottom,
          },
        });
        if (lockViewport && Math.abs(centerDelta.dx) > 1) {
          const zoom = map.getZoom();
          const currentCenter = map.getCenter();
          const currentCenterProjected = map.project(currentCenter, zoom);
          const correctedCenterProjected = L.point(
            currentCenterProjected.x + centerDelta.dx,
            currentCenterProjected.y,
          );
          const correctedCenterLatLng = map.unproject(correctedCenterProjected, zoom);
          const beforeSetViewMidpointPx = map.latLngToContainerPoint(routeMidpointLatLng);
          logFitDebug('fitBounds:post-center-adjust', {
            strategy: 'projected-center-x',
            currentCenter,
            correctedCenterLatLng,
            centerDeltaX: centerDelta.dx,
            targetCenterX: targetCenter.x,
            routeMidpointXBeforeSetView: beforeSetViewMidpointPx.x,
          });
          map.once('moveend', () => {
            const midpointAfter = map.latLngToContainerPoint(routeMidpointLatLng);
            const centerAfterAdjust = map.getCenter();
            logFitDebug('fitBounds:post-center-adjust:moveend', {
              centerAfterAdjust,
              routeMidpointXAfterSetView: midpointAfter.x,
              targetCenterX: targetCenter.x,
              remainingDeltaX: midpointAfter.x - targetCenter.x,
            });
          });
          map.setView(correctedCenterLatLng, zoom, { animate: false });
          const immediateCenter = map.getCenter();
          const immediateMidpoint = map.latLngToContainerPoint(routeMidpointLatLng);
          logFitDebug('fitBounds:post-center-adjust:immediate', {
            requestedCenter: correctedCenterLatLng,
            immediateCenter,
            routeMidpointXImmediate: immediateMidpoint.x,
            targetCenterX: targetCenter.x,
            immediateDeltaX: immediateMidpoint.x - targetCenter.x,
          });
        }
      });
      map.fitBounds(points as LatLngBoundsExpression, {
        animate: !lockViewport,
        duration: lockViewport ? undefined : 0.45,
        paddingTopLeft,
        paddingBottomRight,
      });
      return;
    }
    if (points.length === 1) {
      const [lat, lon] = points[0] as [number, number];
      logFitDebug('flyTo:single-point', {
        lat,
        lon,
        zoomBefore: map.getZoom(),
      });
      map.flyTo([lat, lon], Math.min(map.getZoom(), 11), {
        animate: !lockViewport,
        duration: lockViewport ? undefined : 0.35,
      });
      map.once('moveend', () => {
        logFitDebug('flyTo:after-moveend', {
          zoomAfter: map.getZoom(),
          centerAfter: map.getCenter(),
        });
      });
      return;
    }
    logFitDebug('effect:skip:no-points');
  }, [nonce, origin, destination, managedStop, route?.id, map, lockViewport, logFitDebug]);

  return null;
}

function TutorialGuidePanHandler({
  guideTarget,
  visible,
  viewportLocked,
}: {
  guideTarget: TutorialGuideTarget | null | undefined;
  visible: boolean;
  viewportLocked: boolean;
}) {
  const map = useMap();

  const computeGuidePanShift = useCallback(
    (guideBounds: L.LatLngBounds, guideStage: number) => {
      const mapRect = map.getContainer().getBoundingClientRect();
      const margin = 14;
      const maxShiftX = Math.ceil(mapRect.width * 0.32);
      const maxShiftY = Math.ceil(mapRect.height * 0.26);
      let shiftX = 0;
      let shiftY = 0;

      const nw = map.latLngToContainerPoint(guideBounds.getNorthWest());
      const se = map.latLngToContainerPoint(guideBounds.getSouthEast());
      const guideRect = {
        left: Math.min(nw.x, se.x),
        right: Math.max(nw.x, se.x),
        top: Math.min(nw.y, se.y),
        bottom: Math.max(nw.y, se.y),
      };

      const toContainerRect = (rect: DOMRect) => ({
        left: rect.left - mapRect.left,
        right: rect.right - mapRect.left,
        top: rect.top - mapRect.top,
        bottom: rect.bottom - mapRect.top,
      });

      const applyOccluderShift = (rect: DOMRect) => {
        const occ = toContainerRect(rect);
        const visibleOcc = {
          left: Math.max(0, occ.left),
          right: Math.min(mapRect.width, occ.right),
          top: Math.max(0, occ.top),
          bottom: Math.min(mapRect.height, occ.bottom),
        };

        if (visibleOcc.right <= visibleOcc.left || visibleOcc.bottom <= visibleOcc.top) {
          return;
        }

        const currentRect = {
          left: guideRect.left + shiftX,
          right: guideRect.right + shiftX,
          top: guideRect.top + shiftY,
          bottom: guideRect.bottom + shiftY,
        };

        const intersectX =
          Math.min(currentRect.right, visibleOcc.right) - Math.max(currentRect.left, visibleOcc.left);
        const intersectY =
          Math.min(currentRect.bottom, visibleOcc.bottom) - Math.max(currentRect.top, visibleOcc.top);
        if (intersectX <= 0 || intersectY <= 0) {
          return;
        }

        const centerX = (visibleOcc.left + visibleOcc.right) / 2;
        const centerY = (visibleOcc.top + visibleOcc.bottom) / 2;
        const midX = mapRect.width / 2;
        const midY = mapRect.height / 2;

        if (centerX <= midX) {
          const needed = Math.ceil(Math.max(0, visibleOcc.right - currentRect.left) + margin);
          shiftX = Math.max(shiftX, needed);
        } else {
          const needed = Math.ceil(Math.max(0, currentRect.right - visibleOcc.left) + margin);
          shiftX = Math.min(shiftX, -needed);
        }

        if (centerY <= midY) {
          const needed = Math.ceil(Math.max(0, visibleOcc.bottom - currentRect.top) + margin);
          shiftY = Math.max(shiftY, needed);
        } else {
          const needed = Math.ceil(Math.max(0, currentRect.bottom - visibleOcc.top) + margin);
          shiftY = Math.min(shiftY, -needed);
        }
      };

      const panel = document.querySelector<HTMLElement>('.panel:not(.isCollapsed)');
      if (panel) {
        applyOccluderShift(panel.getBoundingClientRect());
      }

      const tutorialCard = document.querySelector<HTMLElement>(
        '.tutorialOverlay.isRunning .tutorialOverlay__card',
      );
      if (tutorialCard && guideStage !== 1) {
        applyOccluderShift(tutorialCard.getBoundingClientRect());
      }

      shiftX = Math.max(-maxShiftX, Math.min(maxShiftX, shiftX));
      shiftY = Math.max(-maxShiftY, Math.min(maxShiftY, shiftY));

      const shifted = {
        left: guideRect.left + shiftX,
        right: guideRect.right + shiftX,
        top: guideRect.top + shiftY,
        bottom: guideRect.bottom + shiftY,
      };

      if (shifted.left < margin) {
        shiftX += margin - shifted.left;
      } else if (shifted.right > mapRect.width - margin) {
        shiftX -= shifted.right - (mapRect.width - margin);
      }

      if (shifted.top < margin) {
        shiftY += margin - shifted.top;
      } else if (shifted.bottom > mapRect.height - margin) {
        shiftY -= shifted.bottom - (mapRect.height - margin);
      }

      return [Math.round(shiftX), Math.round(shiftY)] as const;
    },
    [map],
  );

  useEffect(() => {
    if (!visible || !guideTarget) return;
    if (guideTarget.stage === 3) {
      // Midpoint drag guidance should not alter the current viewport.
      return;
    }
    let disposed = false;
    let panAdjustRaf = 0;

    const applyGuideViewport = () => {
      const radiusMeters = Math.max(1, guideTarget.radius_km * 1000);
      const earthRadiusMeters = 6371000;
      const latRad = (guideTarget.lat * Math.PI) / 180;
      const deltaLatDeg = (radiusMeters / earthRadiusMeters) * (180 / Math.PI);
      const cosLat = Math.max(0.01, Math.abs(Math.cos(latRad)));
      const deltaLonDeg =
        (radiusMeters / (earthRadiusMeters * cosLat)) * (180 / Math.PI);
      const south = Math.max(-90, guideTarget.lat - deltaLatDeg);
      const north = Math.min(90, guideTarget.lat + deltaLatDeg);
      const west = guideTarget.lon - deltaLonDeg;
      const east = guideTarget.lon + deltaLonDeg;
      const guideBounds = L.latLngBounds([south, west], [north, east]);

      map.stop();
      map.fitBounds(guideBounds, {
        animate: !viewportLocked,
        duration: viewportLocked ? undefined : 0.45,
        paddingTopLeft: [22, 22],
        paddingBottomRight: [22, 22],
        maxZoom: guideTarget.zoom,
      });

      if (panAdjustRaf) {
        window.cancelAnimationFrame(panAdjustRaf);
      }
      panAdjustRaf = window.requestAnimationFrame(() => {
        if (disposed) return;
        const [shiftX, shiftY] = computeGuidePanShift(guideBounds, guideTarget.stage);
        if (Math.abs(shiftX) < 1 && Math.abs(shiftY) < 1) {
          return;
        }
        map.panBy([shiftX, shiftY], {
          animate: !viewportLocked,
          duration: viewportLocked ? undefined : 0.35,
        });
      });
    };

    applyGuideViewport();

    let raf = 0;
    const handleResize = () => {
      if (raf) {
        window.cancelAnimationFrame(raf);
      }
      raf = window.requestAnimationFrame(() => {
        applyGuideViewport();
      });
    };

    window.addEventListener('resize', handleResize, { passive: true });
    window.addEventListener('orientationchange', handleResize);

    return () => {
      disposed = true;
      if (panAdjustRaf) {
        window.cancelAnimationFrame(panAdjustRaf);
      }
      if (raf) {
        window.cancelAnimationFrame(raf);
      }
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleResize);
    };
  }, [
    computeGuidePanShift,
    guideTarget?.lat,
    guideTarget?.lon,
    guideTarget?.pan_nonce,
    guideTarget?.radius_km,
    guideTarget?.zoom,
    visible,
    map,
    viewportLocked,
  ]);

  return null;
}

function MapInteractionLockHandler({
  viewportLocked,
}: {
  viewportLocked: boolean;
}) {
  const map = useMap();

  useEffect(() => {
    const mapWithTap = map as L.Map & {
      tap?: {
        enable: () => void;
        disable: () => void;
      };
    };

    const enableInteractions = () => {
      map.dragging.enable();
      map.scrollWheelZoom.enable();
      map.doubleClickZoom.enable();
      map.boxZoom.enable();
      map.keyboard.enable();
      map.touchZoom.enable();
      mapWithTap.tap?.enable();
    };

    const disableInteractions = () => {
      map.dragging.disable();
      map.scrollWheelZoom.disable();
      map.doubleClickZoom.disable();
      map.boxZoom.disable();
      map.keyboard.disable();
      map.touchZoom.disable();
      mapWithTap.tap?.disable();
    };

    if (!viewportLocked) {
      enableInteractions();
      return;
    }

    disableInteractions();

    return () => {
      enableInteractions();
    };
  }, [map, viewportLocked]);

  return null;
}

function MapBoundsModeHandler({
  relaxBounds,
}: {
  relaxBounds: boolean;
}) {
  const map = useMap();

  useEffect(() => {
    if (relaxBounds) {
      map.options.maxBoundsViscosity = 0;
      map.setMaxBounds(WORLD_BOUNDS);
      return;
    }
    map.options.maxBoundsViscosity = 1.0;
    map.setMaxBounds(UK_BOUNDS);
  }, [map, relaxBounds]);

  return null;
}

function ZoomControlVisibilityHandler({
  hideZoomControls,
}: {
  hideZoomControls: boolean;
}) {
  const map = useMap();

  useEffect(() => {
    const control = map.zoomControl as L.Control.Zoom | undefined;
    const controlContainer = control?.getContainer() ?? null;
    const fallbackContainer = map.getContainer().querySelector<HTMLElement>('.leaflet-control-zoom');
    const container = controlContainer ?? fallbackContainer;

    if (hideZoomControls) {
      if (container) {
        container.style.display = 'none';
        container.style.pointerEvents = 'none';
        container.setAttribute('aria-hidden', 'true');
      }
      if (controlContainer && control && controlContainer.parentElement) {
        map.removeControl(control);
      }
      return;
    }

    if (controlContainer && control && !controlContainer.parentElement) {
      map.addControl(control);
    }
    const visibleContainer = control?.getContainer() ?? fallbackContainer;
    if (visibleContainer) {
      visibleContainer.style.display = '';
      visibleContainer.style.pointerEvents = '';
      visibleContainer.removeAttribute('aria-hidden');
    }
  }, [hideZoomControls, map]);

  return null;
}

function fmtCoord(v: number) {
  return Number.isFinite(v) ? v.toFixed(5) : String(v);
}

function CloseIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 6L6 18" />
      <path d="M6 6l12 12" />
    </svg>
  );
}

function SwapIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M7 7h13" />
      <path d="M17 3l3 4-3 4" />
      <path d="M17 17H4" />
      <path d="M7 21l-3-4 3-4" />
    </svg>
  );
}

function CopyIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M9 9h10v12H9z" />
      <path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1" />
    </svg>
  );
}

async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for older browsers / permission issues.
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      ta.style.pointerEvents = 'none';
      document.body.appendChild(ta);
      ta.focus();
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      return true;
    } catch {
      return false;
    }
  }
}

function downsamplePolyline(
  coords: [number, number][],
  maxPoints: number = MAX_POLYLINE_POINTS,
): [number, number][] {
  if (maxPoints < 2 || coords.length <= maxPoints) return coords;

  const lastIdx = coords.length - 1;
  const stride = lastIdx / (maxPoints - 1);
  const out: [number, number][] = [coords[0]];
  let prevIdx = 0;

  for (let i = 1; i < maxPoints - 1; i += 1) {
    let idx = Math.round(i * stride);
    idx = Math.min(lastIdx - 1, Math.max(prevIdx + 1, idx));
    out.push(coords[idx]);
    prevIdx = idx;
  }

  out.push(coords[lastIdx]);
  return out;
}

function incidentPalette(eventType: IncidentEventType): {
  stroke: string;
  fill: string;
} {
  if (eventType === 'closure') {
    return {
      stroke: 'rgba(239, 68, 68, 0.95)',
      fill: 'rgba(239, 68, 68, 0.84)',
    };
  }
  if (eventType === 'accident') {
    return {
      stroke: 'rgba(249, 115, 22, 0.95)',
      fill: 'rgba(249, 115, 22, 0.84)',
    };
  }
  return {
    stroke: 'rgba(245, 158, 11, 0.95)',
    fill: 'rgba(245, 158, 11, 0.84)',
  };
}

function parseHexColor(hex: string): [number, number, number] {
  const raw = hex.trim().replace('#', '');
  if (raw.length !== 6) return [14, 165, 233];
  const r = Number.parseInt(raw.slice(0, 2), 16);
  const g = Number.parseInt(raw.slice(2, 4), 16);
  const b = Number.parseInt(raw.slice(4, 6), 16);
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) {
    return [14, 165, 233];
  }
  return [r, g, b];
}

function darkenHexColor(hex: string, factor = 0.78): string {
  const [r, g, b] = parseHexColor(hex);
  const toHex = (value: number) => Math.max(0, Math.min(255, Math.round(value))).toString(16).padStart(2, '0');
  return `#${toHex(r * factor)}${toHex(g * factor)}${toHex(b * factor)}`;
}

function rgbaFromHex(hex: string, alpha: number): string {
  const [r, g, b] = parseHexColor(hex);
  const clampedAlpha = Math.max(0, Math.min(1, alpha));
  return `rgba(${r}, ${g}, ${b}, ${clampedAlpha})`;
}

function makeDutyStopIcon(sequence: number, colorHex: string, selected = false) {
  const darkHex = darkenHexColor(colorHex, 0.72);
  const shadowColor = rgbaFromHex(colorHex, 0.48);
  const ringColor = rgbaFromHex(colorHex, 0.24);
  return L.divIcon({
    className: `dutyStopPin ${selected ? 'isSelected' : ''}`.trim(),
    html: `<span class="dutyStopPin__label" style="--stop-pin-color:${colorHex}; --stop-pin-color-deep:${darkHex}; --stop-pin-shadow:${shadowColor}; --stop-pin-ring:${ringColor};">${sequence}</span>`,
    iconSize: [30, 30],
    iconAnchor: [15, 15],
    popupAnchor: [0, -14],
  });
}

function PlusIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="18"
      height="18"
      fill="none"
      stroke="currentColor"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 5v14" />
      <path d="M5 12h14" />
    </svg>
  );
}

function SaveIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <path d="M17 21v-8H7v8" />
      <path d="M7 3v5h8" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 24 24"
      width="16"
      height="16"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M6 6l1 14h10l1-14" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
    </svg>
  );
}

export default function MapView({
  origin,
  destination,
  tutorialDraftOrigin = null,
  tutorialDraftDestination = null,
  tutorialDragDraftOrigin = null,
  tutorialDragDraftDestination = null,
  managedStop = null,
  originLabel = 'Start',
  destinationLabel = 'End',
  selectedPinId = null,
  focusPinRequest = null,
  fitAllRequestNonce = 0,
  route,
  failureOverlay = null,
  timeLapsePosition,
  dutyStops = [],
  showStopOverlay = true,
  showIncidentOverlay = true,
  showSegmentTooltips = true,
  showPreviewConnector = true,
  overlayLabels,
  onTutorialAction,
  onTutorialTargetState,
  tutorialMapLocked = false,
  tutorialMapDimmed = false,
  tutorialViewportLocked = false,
  tutorialHideZoomControls = false,
  tutorialRelaxBounds = false,
  tutorialExpectedAction = null,
  tutorialGuideTarget = null,
  tutorialGuideVisible = false,
  tutorialConfirmPin = null,
  tutorialDragConfirmPin = null,
  onMapClick,
  onSelectPinId,
  onMoveMarker,
  onMoveStop,
  onAddStopFromPin,
  onRenameStop,
  onDeleteStop,
  onFocusPin,
  onSwapMarkers,
  onTutorialConfirmPin,
  onTutorialConfirmDrag,
}: Props) {
  const initialCenterRef = useRef<LatLngExpression>(
    origin ? ([origin.lat, origin.lon] as [number, number]) : ([52.4862, -1.8904] as [number, number]),
  );
  const originRef = useRef<L.Marker>(null);
  const destRef = useRef<L.Marker>(null);
  const stopRef = useRef<L.Marker>(null);

  const [copied, setCopied] = useState<MarkerKind | 'stop' | null>(null);
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null);
  const [stopDraftLabel, setStopDraftLabel] = useState(managedStop?.label ?? 'Stop #1');
  const [draggingPinId, setDraggingPinId] = useState<PinSelectionId | null>(null);
  const suppressMapClickUntilRef = useRef(0);
  const suppressMarkerClickUntilRef = useRef(0);
  const mapPaneRef = useRef<HTMLDivElement | null>(null);
  const tutorialPulseMapLogSeqRef = useRef(0);
  const [dragPreview, setDragPreview] = useState<{
    origin: LatLng | null;
    destination: LatLng | null;
    stop: LatLng | null;
  }>({
    origin: null,
    destination: null,
    stop: null,
  });
  const logMapDim = useCallback(
    (_event: string, _payload?: Record<string, unknown>) => {},
    [],
  );
  const logTutorialPulseMap = useCallback((event: string, payload?: Record<string, unknown>) => {
    if (typeof window === 'undefined') return;
    tutorialPulseMapLogSeqRef.current += 1;
    const elapsed = window.performance.now().toFixed(1);
    if (payload) {
      console.log(
        `[tutorial-pulse-map-debug][${tutorialPulseMapLogSeqRef.current}] +${elapsed}ms ${event}`,
        payload,
      );
      return;
    }
    console.log(
      `[tutorial-pulse-map-debug][${tutorialPulseMapLogSeqRef.current}] +${elapsed}ms ${event}`,
    );
  }, []);

  useEffect(() => {
    if (!tutorialMapLocked && !tutorialMapDimmed && !tutorialViewportLocked) return;
    if (typeof window === 'undefined') return;

    const pane = mapPaneRef.current;
    const stage = pane?.closest('.mapStage') as HTMLElement | null;
    const overlay = document.querySelector<HTMLElement>('.tutorialOverlay');
    const backdrop = overlay?.querySelector<HTMLElement>('.tutorialOverlay__backdrop') ?? null;
    const spotlight = overlay?.querySelector<HTMLElement>('.tutorialOverlay__spotlight') ?? null;
    const paneStyle = pane ? window.getComputedStyle(pane) : null;
    const stageAfterStyle = stage ? window.getComputedStyle(stage, '::after') : null;
    const backdropStyle = backdrop ? window.getComputedStyle(backdrop) : null;

    logMapDim('map-pane-style-snapshot', {
      tutorialMapLocked,
      tutorialMapDimmed,
      tutorialViewportLocked,
      tutorialHideZoomControls,
      tutorialGuideVisible,
      tutorialExpectedAction,
      mapPaneClass: pane?.className ?? null,
      mapStageClass: stage?.className ?? null,
      overlayClass: overlay?.className ?? null,
      hasSpotlight: Boolean(spotlight),
      mapPaneFilter: paneStyle?.filter ?? null,
      mapPaneOpacity: paneStyle?.opacity ?? null,
      mapStageAfterBackground: stageAfterStyle?.backgroundColor ?? null,
      mapStageAfterContent: stageAfterStyle?.content ?? null,
      backdropBackground: backdropStyle?.backgroundColor ?? null,
      backdropDisplay: backdropStyle?.display ?? null,
      backdropOpacity: backdropStyle?.opacity ?? null,
    });
  }, [
    logMapDim,
    tutorialExpectedAction,
    tutorialGuideVisible,
    tutorialHideZoomControls,
    tutorialMapDimmed,
    tutorialMapLocked,
    tutorialViewportLocked,
  ]);

  useEffect(() => {
    if (!tutorialExpectedAction || !tutorialExpectedAction.includes('popup_close')) return;
    const pane = mapPaneRef.current;
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const closeButtons = Array.from(
      document.querySelectorAll<HTMLElement>(
        '[data-tutorial-action="map.popup_close_origin_marker"], [data-tutorial-action="map.popup_close_destination_marker"], [data-tutorial-action="map.popup_close"]',
      ),
    );
    logTutorialPulseMap('popup-close-step:dom-snapshot', {
      tutorialExpectedAction,
      selectedPinId,
      originPopupOpen: originRef.current?.isPopupOpen?.() ?? null,
      destinationPopupOpen: destRef.current?.isPopupOpen?.() ?? null,
      stopPopupOpen: stopRef.current?.isPopupOpen?.() ?? null,
      reducedMotion,
      mapPaneClass: pane?.className ?? null,
      closeButtonCount: closeButtons.length,
      closeButtons: closeButtons.map((node) => {
        const styles = window.getComputedStyle(node);
        const rect = node.getBoundingClientRect();
        return {
          actionId: node.dataset.tutorialAction ?? '',
          className: node.className,
          dataTutorialState: node.getAttribute('data-tutorial-state'),
          animationName: styles.animationName,
          animationDuration: styles.animationDuration,
          animationPlayState: styles.animationPlayState,
          boxShadow: styles.boxShadow,
          opacity: styles.opacity,
          filter: styles.filter,
          pointerEvents: styles.pointerEvents,
          rect: {
            x: Math.round(rect.x),
            y: Math.round(rect.y),
            width: Math.round(rect.width),
            height: Math.round(rect.height),
          },
        };
      }),
    });
    for (const node of closeButtons) {
      const styles = window.getComputedStyle(node);
      console.log(
        `[tutorial-pulse-map-debug] close-button action=${node.dataset.tutorialAction ?? ''} reducedMotion=${String(reducedMotion)} anim=${styles.animationName} duration=${styles.animationDuration} iter=${styles.animationIterationCount} play=${styles.animationPlayState} class=${node.className}`,
      );
    }
  }, [logTutorialPulseMap, selectedPinId, tutorialExpectedAction]);

  useEffect(() => {
    if (!copied) return;
    const t = window.setTimeout(() => setCopied(null), 900);
    return () => window.clearTimeout(t);
  }, [copied]);

  useEffect(() => {
    setStopDraftLabel(managedStop?.label ?? 'Stop #1');
  }, [managedStop?.id, managedStop?.label]);

  useEffect(() => {
    setActiveSegmentId(null);
  }, [route?.id, showSegmentTooltips]);

  useEffect(() => {
    if (dragPreview.origin && origin) {
      const same =
        Math.abs(dragPreview.origin.lat - origin.lat) <= 1e-6 &&
        Math.abs(dragPreview.origin.lon - origin.lon) <= 1e-6;
      if (same) {
        setDragPreview((prev) => ({ ...prev, origin: null }));
        if (draggingPinId === 'origin') setDraggingPinId(null);
      }
    }
  }, [origin, dragPreview.origin, draggingPinId]);

  useEffect(() => {
    if (dragPreview.destination && destination) {
      const same =
        Math.abs(dragPreview.destination.lat - destination.lat) <= 1e-6 &&
        Math.abs(dragPreview.destination.lon - destination.lon) <= 1e-6;
      if (same) {
        setDragPreview((prev) => ({ ...prev, destination: null }));
        if (draggingPinId === 'destination') setDraggingPinId(null);
      }
    }
  }, [destination, dragPreview.destination, draggingPinId]);

  useEffect(() => {
    if (dragPreview.stop && managedStop) {
      const same =
        Math.abs(dragPreview.stop.lat - managedStop.lat) <= 1e-6 &&
        Math.abs(dragPreview.stop.lon - managedStop.lon) <= 1e-6;
      if (same) {
        setDragPreview((prev) => ({ ...prev, stop: null }));
        if (draggingPinId === 'stop-1') setDraggingPinId(null);
      }
    }
  }, [managedStop, dragPreview.stop, draggingPinId]);

  useEffect(() => {
    if (selectedPinId === null && draggingPinId === null) {
      setDraggingPinId(null);
      setDragPreview({ origin: null, destination: null, stop: null });
    }
  }, [selectedPinId, draggingPinId]);

  const suppressInteractionsBriefly = useCallback(() => {
    const until = Date.now() + 1400;
    suppressMapClickUntilRef.current = until;
    suppressMarkerClickUntilRef.current = until;
  }, []);

  useEffect(() => {
    if (draggingPinId !== null) return;
    try {
      if (originRef.current) {
        if (selectedPinId === 'origin') originRef.current.openPopup();
        else originRef.current.closePopup();
      }
      if (destRef.current) {
        if (selectedPinId === 'destination') destRef.current.openPopup();
        else destRef.current.closePopup();
      }
      if (stopRef.current) {
        if (selectedPinId === 'stop-1') stopRef.current.openPopup();
        else stopRef.current.closePopup();
      }
    } catch {
      // no-op
    }
  }, [selectedPinId, draggingPinId]);

  const renderedOrigin = tutorialDragDraftOrigin ?? tutorialDraftOrigin ?? origin;
  const renderedDestination = tutorialDragDraftDestination ?? tutorialDraftDestination ?? destination;
  const effectiveOrigin = dragPreview.origin ?? renderedOrigin;
  const effectiveDestination = dragPreview.destination ?? renderedDestination;
  const effectiveStop = dragPreview.stop ?? (managedStop ? { lat: managedStop.lat, lon: managedStop.lon } : null);
  const stopDisplayName = managedStop?.label?.trim() || 'Stop #1';
  const effectiveDutyStops = useMemo(() => {
    if (effectiveStop) {
      return [{ lat: effectiveStop.lat, lon: effectiveStop.lon, label: stopDisplayName }];
    }
    return dutyStops;
  }, [effectiveStop, stopDisplayName, dutyStops]);
  const previewNodes = useMemo(
    () => buildPreviewRouteNodes(effectiveOrigin, effectiveDestination, effectiveDutyStops),
    [effectiveOrigin, effectiveDestination, effectiveDutyStops],
  );
  const stopNodeColor = useMemo(
    () => previewNodes.find((node) => node.id === 'stop-1')?.color ?? '#0EA5E9',
    [previewNodes],
  );

  const originIcon = useMemo(
    () => makePinIcon('origin', selectedPinId === 'origin'),
    [selectedPinId],
  );
  const destIcon = useMemo(
    () => makePinIcon('destination', selectedPinId === 'destination'),
    [selectedPinId],
  );
  const stopIcon = useMemo(
    () => makeDutyStopIcon(1, stopNodeColor, selectedPinId === 'stop-1'),
    [selectedPinId, stopNodeColor],
  );

  const polylinePositions: LatLngExpression[] = useMemo(() => {
    const coords = route?.geometry?.coordinates ?? [];
    const slim = downsamplePolyline(coords);
    return slim.map(([lon, lat]) => [lat, lon] as [number, number]);
  }, [route]);
  const failureLinePositions: LatLngExpression[] = useMemo(() => {
    if (!failureOverlay || !effectiveOrigin || !effectiveDestination) return [];
    return [
      [effectiveOrigin.lat, effectiveOrigin.lon] as [number, number],
      [effectiveDestination.lat, effectiveDestination.lon] as [number, number],
    ];
  }, [effectiveDestination, effectiveOrigin, failureOverlay]);
  const failureMidpoint = useMemo(() => {
    if (!failureOverlay || !effectiveOrigin || !effectiveDestination) return null;
    return {
      lat: (effectiveOrigin.lat + effectiveDestination.lat) / 2,
      lon: (effectiveOrigin.lon + effectiveDestination.lon) / 2,
    };
  }, [effectiveDestination, effectiveOrigin, failureOverlay]);
  const failureOverlayActive = Boolean(failureOverlay && failureLinePositions.length >= 2);

  const stopOverlayPoints = useMemo(() => {
    if (!showStopOverlay) return [];
    return buildStopOverlayPoints(effectiveOrigin, effectiveDestination, effectiveDutyStops);
  }, [showStopOverlay, effectiveOrigin, effectiveDestination, effectiveDutyStops]);
  const previewDotSegments = useMemo(
    () => buildPreviewRouteSegmentsFromNodes(previewNodes),
    [previewNodes],
  );

  useEffect(() => {
    const midpointStepLike =
      tutorialExpectedAction === 'pins.add_stop' ||
      tutorialExpectedAction === 'map.add_stop_midpoint' ||
      tutorialExpectedAction === 'map.click_stop_marker' ||
      tutorialExpectedAction === 'map.drag_stop_marker' ||
      Boolean(managedStop);
    if (!midpointStepLike) return;
    logMapDim('midpoint-visual-state', {
      tutorialExpectedAction,
      tutorialMapLocked,
      tutorialMapDimmed,
      tutorialViewportLocked,
      showStopOverlay,
      hasManagedStop: Boolean(managedStop),
      managedStop,
      hasEffectiveStop: Boolean(effectiveStop),
      effectiveStop,
      selectedPinId,
      draggingPinId,
      hasDragPreviewStop: Boolean(dragPreview.stop),
      dragPreviewStop: dragPreview.stop,
      dutyStopsCount: dutyStops.length,
      effectiveDutyStopsCount: effectiveDutyStops.length,
      previewNodes: previewNodes.map((node) => ({
        id: node.id,
        role: node.role,
        lat: node.lat,
        lon: node.lon,
      })),
      previewDotSegmentsCount: previewDotSegments.length,
    });
  }, [
    draggingPinId,
    dragPreview.stop,
    dutyStops.length,
    effectiveDutyStops,
    effectiveStop,
    logMapDim,
    managedStop,
    previewDotSegments.length,
    previewNodes,
    selectedPinId,
    showStopOverlay,
    tutorialExpectedAction,
    tutorialMapDimmed,
    tutorialMapLocked,
    tutorialViewportLocked,
  ]);

  const incidentOverlayPoints = useMemo(() => {
    if (!showIncidentOverlay || !route) return [];
    return buildIncidentOverlayPoints(route);
  }, [showIncidentOverlay, route]);
  const segmentBuckets = useMemo(() => {
    if (!showSegmentTooltips || !route) return [];
    return buildSegmentBuckets(route, 120);
  }, [showSegmentTooltips, route]);

  useEffect(() => {
    onTutorialTargetState?.({
      hasSegmentTooltipPath: segmentBuckets.length > 0,
      hasIncidentMarkers: incidentOverlayPoints.length > 0,
    });
  }, [incidentOverlayPoints.length, onTutorialTargetState, segmentBuckets.length]);

  // Keep an immutable initial center so route/pin state updates never recenter the map unexpectedly.
  const center = initialCenterRef.current;

  const suppressRoutePath = draggingPinId !== null;
  const isActionExpected = useCallback(
    (actionId: string) => {
      if (!tutorialExpectedAction) return true;
      return tutorialExpectedAction === actionId;
    },
    [tutorialExpectedAction],
  );
  const isMarkerActionAllowed = useCallback(
    (
      actionId: string,
      markerKind?: 'origin' | 'destination',
    ) => {
      if (
        markerKind === 'origin' &&
        tutorialExpectedAction === 'map.confirm_drag_origin_marker' &&
        actionId === 'map.drag_origin_marker'
      ) {
        return true;
      }
      if (
        markerKind === 'destination' &&
        tutorialExpectedAction === 'map.confirm_drag_destination_marker' &&
        actionId === 'map.drag_destination_marker'
      ) {
        return true;
      }
      return isActionExpected(actionId);
    },
    [isActionExpected, tutorialExpectedAction],
  );
  const canUsePopupSwap = isActionExpected('map.popup_swap');
  const canUsePopupAddStop = isActionExpected('map.add_stop_midpoint');
  const disableTutorialPopupAutoPan = Boolean(
    tutorialMapLocked || tutorialViewportLocked || tutorialGuideVisible || tutorialExpectedAction !== null,
  );
  const mapInteractionLocked = tutorialMapLocked || tutorialViewportLocked;
  const blockMarkerClick = useCallback(
    (event: {
      originalEvent?: MouseEvent;
      target?: L.Layer;
    }) => {
      event.originalEvent?.preventDefault();
      event.originalEvent?.stopPropagation();
      try {
        (event.target as L.Marker | undefined)?.closePopup();
      } catch {
        // no-op
      }
    },
    [],
  );

  const handleCopyCoords = useCallback(async (kind: MarkerKind | 'stop', lat: number, lon: number) => {
    const text = `${fmtCoord(lat)}, ${fmtCoord(lon)}`;
    const ok = await copyToClipboard(text);
    if (ok) {
      setCopied(kind);
      onTutorialAction?.('map.popup_copy');
    }
  }, [onTutorialAction]);
  const resolvePopupCloseAction = useCallback(() => {
    if (tutorialExpectedAction === 'map.popup_close_destination_marker') {
      return 'map.popup_close_destination_marker';
    }
    if (tutorialExpectedAction === 'map.popup_close_origin_marker') {
      return 'map.popup_close_origin_marker';
    }
    return 'map.popup_close';
  }, [tutorialExpectedAction]);

  const handleMapClickFromMap = useCallback(
    (lat: number, lon: number) => {
      if (tutorialMapLocked) return;
      if (draggingPinId !== null) return;
      const now = Date.now();
      if (now < suppressMapClickUntilRef.current) return;
      onMapClick(lat, lon);
    },
    [draggingPinId, onMapClick, tutorialMapLocked],
  );

  return (
    <div
      ref={mapPaneRef}
      className={`mapPane ${tutorialMapDimmed ? 'mapPane--tutorialLocked' : ''} ${tutorialGuideVisible ? 'mapPane--guide' : ''}`.trim()}
      data-segment-tooltips={showSegmentTooltips ? 'on' : 'off'}
      data-tutorial-hide-zoom-controls={tutorialHideZoomControls ? 'true' : 'false'}
      data-tutorial-id="map.interactive"
    >
      <MapContainer
        key={MAP_HOT_RELOAD_KEY}
        center={center}
        zoom={11}
        minZoom={5}
        maxZoom={18}
        zoomControl={!tutorialHideZoomControls}
        zoomAnimation={!tutorialGuideVisible}
        fadeAnimation={!tutorialGuideVisible}
        markerZoomAnimation={!tutorialGuideVisible}
        dragging={!mapInteractionLocked}
        scrollWheelZoom={!mapInteractionLocked}
        doubleClickZoom={!mapInteractionLocked}
        boxZoom={!mapInteractionLocked}
        keyboard={!mapInteractionLocked}
        touchZoom={!mapInteractionLocked}
        maxBounds={tutorialRelaxBounds ? WORLD_BOUNDS : UK_BOUNDS}
        maxBoundsViscosity={tutorialRelaxBounds ? 0 : 1.0}
        style={{ height: '100%', width: '100%' }}
      >
        <MapBoundsModeHandler relaxBounds={tutorialRelaxBounds} />
        <FocusPinRequestHandler
          request={focusPinRequest}
          origin={renderedOrigin}
          destination={renderedDestination}
          managedStop={managedStop}
          originRef={originRef}
          destRef={destRef}
          stopRef={stopRef}
        />
        <FitAllRequestHandler
          nonce={fitAllRequestNonce}
          origin={renderedOrigin}
          destination={renderedDestination}
          managedStop={managedStop}
          route={route}
          lockViewport={mapInteractionLocked}
        />
        <TutorialGuidePanHandler
          guideTarget={tutorialGuideTarget}
          visible={tutorialGuideVisible}
          viewportLocked={tutorialViewportLocked}
        />
        <MapInteractionLockHandler viewportLocked={tutorialViewportLocked} />
        <ZoomControlVisibilityHandler hideZoomControls={tutorialHideZoomControls} />

        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          subdomains={['a', 'b', 'c', 'd']}
        />

        <ClickHandler onMapClick={handleMapClickFromMap} />

        {tutorialGuideVisible && tutorialGuideTarget ? (
          <>
            <Circle
              center={[tutorialGuideTarget.lat, tutorialGuideTarget.lon]}
              radius={tutorialGuideTarget.radius_km * 1000}
              pathOptions={{
                color: 'rgba(6, 182, 212, 0.94)',
                weight: 2,
                fillColor: 'rgba(6, 182, 212, 0.14)',
                fillOpacity: 0.2,
                dashArray: '6 6',
              }}
              interactive={false}
            />
            <CircleMarker
              center={[tutorialGuideTarget.lat, tutorialGuideTarget.lon]}
              radius={8}
              pathOptions={{
                color: 'rgba(6, 182, 212, 0.95)',
                weight: 2,
                fillColor: 'rgba(6, 182, 212, 0.92)',
                fillOpacity: 0.9,
              }}
              interactive={false}
            >
              <Tooltip permanent={true} direction="top" offset={[0, -10]} className="tutorialGuideTooltip">
                {tutorialGuideTarget.label}
              </Tooltip>
            </CircleMarker>
          </>
        ) : null}

        {renderedOrigin && (
          <Marker
            ref={originRef}
            position={[effectiveOrigin?.lat ?? renderedOrigin.lat, effectiveOrigin?.lon ?? renderedOrigin.lon]}
            icon={originIcon}
            draggable={!tutorialMapLocked && isMarkerActionAllowed('map.drag_origin_marker', 'origin')}
            riseOnHover={true}
            eventHandlers={{
              click(e) {
                const suppressed = Date.now() < suppressMarkerClickUntilRef.current;
                const expectsOriginClose =
                  tutorialExpectedAction === 'map.popup_close_origin_marker';
                const allowOriginClickForClose =
                  expectsOriginClose && (e.target as L.Marker).isPopupOpen();
                if (
                  tutorialMapLocked ||
                  (!isMarkerActionAllowed('map.click_origin_marker', 'origin') &&
                    !allowOriginClickForClose) ||
                  suppressed
                ) {
                  if (suppressed) onSelectPinId?.(null);
                  blockMarkerClick(e);
                  return;
                }
                e.originalEvent?.stopPropagation();
                const marker = e.target as L.Marker;
                const closeActionId = resolvePopupCloseAction();
                if (expectsOriginClose && marker.isPopupOpen()) {
                  logTutorialPulseMap('origin-marker-click-close-path', {
                    tutorialExpectedAction,
                    closeActionId,
                    markerPopupOpen: marker.isPopupOpen(),
                    selectedPinId,
                  });
                  onSelectPinId?.(null);
                  onTutorialAction?.(closeActionId);
                  suppressInteractionsBriefly();
                  marker.closePopup();
                  return;
                }
                onSelectPinId?.(selectedPinId === 'origin' ? null : 'origin');
                onFocusPin?.('origin');
                onTutorialAction?.('map.click_origin_marker');
              },
              dragend(e) {
                if (tutorialMapLocked) return;
                if (!isMarkerActionAllowed('map.drag_origin_marker', 'origin')) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, origin: { lat: pos.lat, lon: pos.lng } }));
                suppressInteractionsBriefly();
                const accepted = onMoveMarker('origin', pos.lat, pos.lng);
                if (!accepted) {
                  setDragPreview((prev) => ({ ...prev, origin: null }));
                }
                if (accepted) {
                  onTutorialAction?.('map.drag_origin_marker');
                }
              },
              dragstart() {
                if (tutorialMapLocked) return;
                if (!isMarkerActionAllowed('map.drag_origin_marker', 'origin')) return;
                suppressInteractionsBriefly();
                onSelectPinId?.(null);
                try {
                  originRef.current?.closePopup();
                } catch {
                  // no-op
                }
                setDraggingPinId('origin');
              },
              drag(e) {
                if (tutorialMapLocked) return;
                if (!isMarkerActionAllowed('map.drag_origin_marker', 'origin')) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, origin: { lat: pos.lat, lon: pos.lng } }));
              },
            }}
          >
            <Popup
              className="markerPopup"
              closeButton={false}
              autoPan={!disableTutorialPopupAutoPan}
              autoPanPadding={[22, 22]}
              eventHandlers={{
                remove() {
                  if (tutorialExpectedAction === 'map.popup_close_origin_marker') {
                    logTutorialPulseMap('origin-popup-remove-event', {
                      tutorialExpectedAction,
                    });
                    onTutorialAction?.('map.popup_close_origin_marker');
                  }
                },
              }}
            >
              <div className="markerPopup__card" onClick={(e) => e.stopPropagation()}>
                <div className="markerPopup__header">
                  <span className="markerPopup__pill markerPopup__pill--origin">
                    {originLabel || 'Start'}
                  </span>

                  <div className="markerPopup__actions">
                    {onSwapMarkers && destination && (
                      <button
                        type="button"
                        className="markerPopup__iconBtn"
                        disabled={!canUsePopupSwap}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!canUsePopupSwap) return;
                          onSwapMarkers();
                          onTutorialAction?.('map.popup_swap');
                        }}
                        aria-label="Swap Start and End"
                        title="Swap Start and End"
                        data-tutorial-action="map.popup_swap"
                      >
                        <SwapIcon />
                      </button>
                    )}

                    {onAddStopFromPin && origin && destination ? (
                      <button
                        type="button"
                        className="markerPopup__iconBtn"
                        disabled={!canUsePopupAddStop}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!canUsePopupAddStop) return;
                          logMapDim('popup-add-stop-click:origin', {
                            tutorialExpectedAction,
                            canUsePopupAddStop,
                            origin,
                            destination,
                            managedStop,
                          });
                          onAddStopFromPin();
                          onTutorialAction?.('map.add_stop_midpoint');
                        }}
                        aria-label="Add stop at midpoint"
                        title="Add stop at midpoint"
                        data-tutorial-action="map.add_stop_midpoint"
                      >
                        <PlusIcon />
                      </button>
                    ) : null}

                    <button
                      type="button"
                      className="markerPopup__iconBtn"
                      onClick={(e) => {
                        e.stopPropagation();
                        logTutorialPulseMap('origin-popup-close-button-click', {
                          tutorialExpectedAction,
                          resolvedActionId: resolvePopupCloseAction(),
                          selectedPinId,
                        });
                        onSelectPinId?.(null);
                        onTutorialAction?.(resolvePopupCloseAction());
                      }}
                      aria-label="Close"
                      title="Close"
                      data-tutorial-action={resolvePopupCloseAction()}
                    >
                      <CloseIcon />
                    </button>
                  </div>
                </div>

                <div className="markerPopup__coordsRow">
                  <button
                    type="button"
                    className="markerPopup__coordsBtn"
                    onClick={(e) => {
                        e.stopPropagation();
                        handleCopyCoords(
                          'origin',
                          effectiveOrigin?.lat ?? renderedOrigin.lat,
                          effectiveOrigin?.lon ?? renderedOrigin.lon,
                        );
                      }}
                    aria-label="Copy coordinates"
                    title="Copy coordinates"
                    data-tutorial-action="map.popup_copy"
                  >
                    <span className="markerPopup__coordsText">
                      {fmtCoord(effectiveOrigin?.lat ?? renderedOrigin.lat)}, {fmtCoord(effectiveOrigin?.lon ?? renderedOrigin.lon)}
                    </span>
                    <span className="markerPopup__coordsIcon">
                      <CopyIcon />
                    </span>
                  </button>

                  {copied === 'origin' && <span className="markerPopup__toast">Copied</span>}
                </div>

                <div className="markerPopup__tinyHint">
                  Start pin cannot be removed individually. Use Clear pins in Setup to reset both pins.
                </div>

              </div>
            </Popup>
            {tutorialConfirmPin === 'origin' && onTutorialConfirmPin ? (
              <Tooltip
                permanent={true}
                interactive={true}
                direction="top"
                offset={[0, -62]}
                className="tutorialConfirmPinTooltip tutorialConfirmPinTooltip--origin"
              >
                <div
                  className="tutorialConfirmPinTooltip__wrap"
                  role="group"
                  aria-label="Confirm Start placement"
                >
                  <button
                    type="button"
                    className="tutorialConfirmPinTooltip__button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      suppressInteractionsBriefly();
                      onTutorialConfirmPin('origin');
                    }}
                    data-tutorial-action="map.confirm_origin_newcastle"
                  >
                    <span className="tutorialConfirmPinTooltip__dot" aria-hidden="true" />
                    Confirm Start
                  </button>
                </div>
              </Tooltip>
            ) : null}
            {tutorialDragConfirmPin === 'origin' && onTutorialConfirmDrag ? (
              <Tooltip
                permanent={true}
                interactive={true}
                direction="top"
                offset={[0, -62]}
                className="tutorialConfirmPinTooltip tutorialConfirmPinTooltip--origin"
              >
                <div
                  className="tutorialConfirmPinTooltip__wrap"
                  role="group"
                  aria-label="Confirm Start drag"
                >
                  <button
                    type="button"
                    className="tutorialConfirmPinTooltip__button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      suppressInteractionsBriefly();
                      onTutorialConfirmDrag('origin');
                    }}
                    data-tutorial-action="map.confirm_drag_origin_marker"
                  >
                    <span className="tutorialConfirmPinTooltip__dot" aria-hidden="true" />
                    Confirm Start Drag
                  </button>
                </div>
              </Tooltip>
            ) : null}
          </Marker>
        )}

        {renderedDestination && (
          <Marker
            ref={destRef}
            position={[
              effectiveDestination?.lat ?? renderedDestination.lat,
              effectiveDestination?.lon ?? renderedDestination.lon,
            ]}
            icon={destIcon}
            draggable={
              !tutorialMapLocked &&
              isMarkerActionAllowed('map.drag_destination_marker', 'destination')
            }
            riseOnHover={true}
            eventHandlers={{
              click(e) {
                const suppressed = Date.now() < suppressMarkerClickUntilRef.current;
                const expectsDestinationClose =
                  tutorialExpectedAction === 'map.popup_close_destination_marker';
                const allowDestinationClickForClose =
                  expectsDestinationClose && (e.target as L.Marker).isPopupOpen();
                if (
                  tutorialMapLocked ||
                  (!isMarkerActionAllowed('map.click_destination_marker', 'destination') &&
                    !allowDestinationClickForClose) ||
                  suppressed
                ) {
                  if (suppressed) onSelectPinId?.(null);
                  blockMarkerClick(e);
                  return;
                }
                e.originalEvent?.stopPropagation();
                const marker = e.target as L.Marker;
                const closeActionId = resolvePopupCloseAction();
                if (expectsDestinationClose && marker.isPopupOpen()) {
                  logTutorialPulseMap('destination-marker-click-close-path', {
                    tutorialExpectedAction,
                    closeActionId,
                    markerPopupOpen: marker.isPopupOpen(),
                    selectedPinId,
                  });
                  onSelectPinId?.(null);
                  onTutorialAction?.(closeActionId);
                  suppressInteractionsBriefly();
                  marker.closePopup();
                  return;
                }
                onSelectPinId?.(selectedPinId === 'destination' ? null : 'destination');
                onFocusPin?.('destination');
                onTutorialAction?.('map.click_destination_marker');
              },
              dragend(e) {
                if (tutorialMapLocked) return;
                if (!isMarkerActionAllowed('map.drag_destination_marker', 'destination')) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, destination: { lat: pos.lat, lon: pos.lng } }));
                suppressInteractionsBriefly();
                const accepted = onMoveMarker('destination', pos.lat, pos.lng);
                if (!accepted) {
                  setDragPreview((prev) => ({ ...prev, destination: null }));
                }
                if (accepted) {
                  onTutorialAction?.('map.drag_destination_marker');
                }
              },
              dragstart() {
                if (tutorialMapLocked) return;
                if (!isMarkerActionAllowed('map.drag_destination_marker', 'destination')) return;
                suppressInteractionsBriefly();
                onSelectPinId?.(null);
                try {
                  destRef.current?.closePopup();
                } catch {
                  // no-op
                }
                setDraggingPinId('destination');
              },
              drag(e) {
                if (tutorialMapLocked) return;
                if (!isMarkerActionAllowed('map.drag_destination_marker', 'destination')) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, destination: { lat: pos.lat, lon: pos.lng } }));
              },
            }}
          >
            <Popup
              className="markerPopup"
              closeButton={false}
              autoPan={!disableTutorialPopupAutoPan}
              autoPanPadding={[22, 22]}
              eventHandlers={{
                remove() {
                  if (tutorialExpectedAction === 'map.popup_close_destination_marker') {
                    logTutorialPulseMap('destination-popup-remove-event', {
                      tutorialExpectedAction,
                    });
                    onTutorialAction?.('map.popup_close_destination_marker');
                  }
                },
              }}
            >
              <div className="markerPopup__card" onClick={(e) => e.stopPropagation()}>
                <div className="markerPopup__header">
                  <span className="markerPopup__pill markerPopup__pill--destination">
                    {destinationLabel || 'End'}
                  </span>

                  <div className="markerPopup__actions">
                    {onSwapMarkers && origin && (
                      <button
                        type="button"
                        className="markerPopup__iconBtn"
                        disabled={!canUsePopupSwap}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!canUsePopupSwap) return;
                          onSwapMarkers();
                          onTutorialAction?.('map.popup_swap');
                        }}
                        aria-label="Swap Start and End"
                        title="Swap Start and End"
                        data-tutorial-action="map.popup_swap"
                      >
                        <SwapIcon />
                      </button>
                    )}

                    {onAddStopFromPin && origin && destination ? (
                      <button
                        type="button"
                        className="markerPopup__iconBtn"
                        disabled={!canUsePopupAddStop}
                        onClick={(e) => {
                          e.stopPropagation();
                          if (!canUsePopupAddStop) return;
                          logMapDim('popup-add-stop-click:destination', {
                            tutorialExpectedAction,
                            canUsePopupAddStop,
                            origin,
                            destination,
                            managedStop,
                          });
                          onAddStopFromPin();
                          onTutorialAction?.('map.add_stop_midpoint');
                        }}
                        aria-label="Add stop at midpoint"
                        title="Add stop at midpoint"
                        data-tutorial-action="map.add_stop_midpoint"
                      >
                        <PlusIcon />
                      </button>
                    ) : null}

                    <button
                      type="button"
                      className="markerPopup__iconBtn"
                      onClick={(e) => {
                        e.stopPropagation();
                        logTutorialPulseMap('destination-popup-close-button-click', {
                          tutorialExpectedAction,
                          resolvedActionId: resolvePopupCloseAction(),
                          selectedPinId,
                        });
                        onSelectPinId?.(null);
                        onTutorialAction?.(resolvePopupCloseAction());
                      }}
                      aria-label="Close"
                      title="Close"
                      data-tutorial-action={resolvePopupCloseAction()}
                    >
                      <CloseIcon />
                    </button>
                  </div>
                </div>

                <div className="markerPopup__coordsRow">
                  <button
                    type="button"
                    className="markerPopup__coordsBtn"
                    onClick={(e) => {
                        e.stopPropagation();
                        handleCopyCoords(
                          'destination',
                          effectiveDestination?.lat ?? renderedDestination.lat,
                          effectiveDestination?.lon ?? renderedDestination.lon,
                        );
                      }}
                    aria-label="Copy coordinates"
                    title="Copy coordinates"
                    data-tutorial-action="map.popup_copy"
                  >
                    <span className="markerPopup__coordsText">
                      {fmtCoord(effectiveDestination?.lat ?? renderedDestination.lat)}, {fmtCoord(effectiveDestination?.lon ?? renderedDestination.lon)}
                    </span>
                    <span className="markerPopup__coordsIcon">
                      <CopyIcon />
                    </span>
                  </button>

                  {copied === 'destination' && <span className="markerPopup__toast">Copied</span>}
                </div>

                <div className="markerPopup__tinyHint">
                  End pin cannot be removed individually. Use Clear pins in Setup to reset both pins.
                </div>

              </div>
            </Popup>
            {tutorialConfirmPin === 'destination' && onTutorialConfirmPin ? (
              <Tooltip
                permanent={true}
                interactive={true}
                direction="top"
                offset={[0, -62]}
                className="tutorialConfirmPinTooltip tutorialConfirmPinTooltip--destination"
              >
                <div
                  className="tutorialConfirmPinTooltip__wrap"
                  role="group"
                  aria-label="Confirm End placement"
                >
                  <button
                    type="button"
                    className="tutorialConfirmPinTooltip__button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      suppressInteractionsBriefly();
                      onTutorialConfirmPin('destination');
                    }}
                    data-tutorial-action="map.confirm_destination_london"
                  >
                    <span className="tutorialConfirmPinTooltip__dot" aria-hidden="true" />
                    Confirm End
                  </button>
                </div>
              </Tooltip>
            ) : null}
            {tutorialDragConfirmPin === 'destination' && onTutorialConfirmDrag ? (
              <Tooltip
                permanent={true}
                interactive={true}
                direction="top"
                offset={[0, -62]}
                className="tutorialConfirmPinTooltip tutorialConfirmPinTooltip--destination"
              >
                <div
                  className="tutorialConfirmPinTooltip__wrap"
                  role="group"
                  aria-label="Confirm End drag"
                >
                  <button
                    type="button"
                    className="tutorialConfirmPinTooltip__button"
                    onClick={(e) => {
                      e.preventDefault();
                      e.stopPropagation();
                      suppressInteractionsBriefly();
                      onTutorialConfirmDrag('destination');
                    }}
                    data-tutorial-action="map.confirm_drag_destination_marker"
                  >
                    <span className="tutorialConfirmPinTooltip__dot" aria-hidden="true" />
                    Confirm End Drag
                  </button>
                </div>
              </Tooltip>
            ) : null}
          </Marker>
        )}

        {showStopOverlay && managedStop ? (
          <Marker
            ref={stopRef}
            position={[effectiveStop?.lat ?? managedStop.lat, effectiveStop?.lon ?? managedStop.lon]}
            icon={stopIcon}
            draggable={!tutorialMapLocked && isActionExpected('map.drag_stop_marker')}
            riseOnHover={true}
            eventHandlers={{
              click(e) {
                const marker = e.target as L.Marker;
                const suppressed = Date.now() < suppressMarkerClickUntilRef.current;
                const expectsStopClose = tutorialExpectedAction === 'map.popup_close';
                const allowStopClickForClose = expectsStopClose && marker.isPopupOpen();
                if (
                  tutorialMapLocked ||
                  (!isActionExpected('map.click_stop_marker') && !allowStopClickForClose) ||
                  suppressed
                ) {
                  if (suppressed) onSelectPinId?.(null);
                  blockMarkerClick(e);
                  return;
                }
                e.originalEvent?.stopPropagation();
                if (expectsStopClose && marker.isPopupOpen()) {
                  onSelectPinId?.(null);
                  onTutorialAction?.('map.popup_close');
                  suppressInteractionsBriefly();
                  marker.closePopup();
                  return;
                }
                onSelectPinId?.(selectedPinId === 'stop-1' ? null : 'stop-1');
                onFocusPin?.('stop-1');
                onTutorialAction?.('map.click_stop_marker');
              },
              dragend(e) {
                if (tutorialMapLocked) return;
                if (!isActionExpected('map.drag_stop_marker')) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, stop: { lat: pos.lat, lon: pos.lng } }));
                suppressInteractionsBriefly();
                const accepted = onMoveStop ? onMoveStop(pos.lat, pos.lng) : false;
                if (!accepted) {
                  const fallback = effectiveStop ?? managedStop;
                  if (fallback) {
                    marker.setLatLng([fallback.lat, fallback.lon]);
                  }
                  setDragPreview((prev) => ({ ...prev, stop: null }));
                } else {
                  onSelectPinId?.(null);
                  onTutorialAction?.('map.drag_stop_marker');
                }
              },
              dragstart() {
                if (tutorialMapLocked) return;
                if (!isActionExpected('map.drag_stop_marker')) return;
                suppressInteractionsBriefly();
                try {
                  stopRef.current?.closePopup();
                } catch {
                  // no-op
                }
                onSelectPinId?.(null);
                setDraggingPinId('stop-1');
              },
              drag(e) {
                if (tutorialMapLocked) return;
                if (!isActionExpected('map.drag_stop_marker')) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, stop: { lat: pos.lat, lon: pos.lng } }));
              },
            }}
          >
            <Popup
              className="stopOverlayPopup"
              autoPan={!disableTutorialPopupAutoPan}
              autoPanPadding={[22, 22]}
              closeButton={false}
              eventHandlers={{
                remove() {
                  if (tutorialExpectedAction === 'map.popup_close') {
                    onTutorialAction?.('map.popup_close');
                  }
                },
              }}
            >
              <div className="overlayPopup__card stopPopup__card" onClick={(e) => e.stopPropagation()}>
                <div className="markerPopup__header">
                  <span className="markerPopup__pill markerPopup__pill--stop">{stopDisplayName}</span>
                  <div className="markerPopup__actions">
                    {onDeleteStop ? (
                      <button
                        type="button"
                        className="markerPopup__iconBtn markerPopup__iconBtn--danger"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteStop();
                          onSelectPinId?.(null);
                          onTutorialAction?.('map.delete_stop');
                        }}
                        aria-label="Delete stop"
                        title="Delete stop"
                        data-tutorial-action="map.delete_stop"
                      >
                        <TrashIcon />
                      </button>
                    ) : null}
                    <button
                      type="button"
                      className="markerPopup__iconBtn"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectPinId?.(null);
                        onTutorialAction?.('map.popup_close');
                      }}
                      aria-label="Close"
                      title="Close"
                      data-tutorial-action="map.popup_close"
                    >
                      <CloseIcon />
                    </button>
                  </div>
                </div>
                <div className="fieldLabelRow" style={{ marginTop: 0 }}>
                  <div className="fieldLabel">Stop name</div>
                </div>
                <div className="stopPopup__renameRow">
                  <input
                    className="input stopPopup__nameInput"
                    value={stopDraftLabel}
                    onChange={(e) => setStopDraftLabel(e.target.value)}
                    placeholder="Stop #1"
                    aria-label="Stop name"
                  />
                  <button
                    type="button"
                    className="markerPopup__iconBtn"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRenameStop?.(stopDraftLabel);
                      onTutorialAction?.('map.rename_stop');
                    }}
                    aria-label="Save stop name"
                    title="Save stop name"
                    data-tutorial-action="map.rename_stop"
                  >
                    <SaveIcon />
                  </button>
                </div>
                <div className="markerPopup__coordsRow">
                  <button
                    type="button"
                    className="markerPopup__coordsBtn"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCopyCoords(
                        'stop',
                        effectiveStop?.lat ?? managedStop.lat,
                        effectiveStop?.lon ?? managedStop.lon,
                      );
                    }}
                    aria-label="Copy stop coordinates"
                    title="Copy stop coordinates"
                    data-tutorial-action="map.popup_copy"
                  >
                    <span className="markerPopup__coordsText">
                      {fmtCoord(effectiveStop?.lat ?? managedStop.lat)}, {fmtCoord(effectiveStop?.lon ?? managedStop.lon)}
                    </span>
                    <span className="markerPopup__coordsIcon">
                      <CopyIcon />
                    </span>
                  </button>
                  {copied === 'stop' && <span className="markerPopup__toast">Copied</span>}
                </div>
              </div>
            </Popup>
          </Marker>
        ) : null}

        {!suppressRoutePath && !failureOverlayActive && polylinePositions.length > 0 && (
          <>
            <Polyline
              positions={polylinePositions}
              pathOptions={{
                className: 'routeGlow',
                color: 'rgba(6, 182, 212, 0.35)',
                weight: 12,
                opacity: 0.9,
              }}
            />
            <Polyline
              positions={polylinePositions}
              pathOptions={{
                className: 'routePath',
                color: 'rgba(124, 58, 237, 0.95)',
                weight: 5,
                opacity: 0.95,
                lineCap: 'round',
                lineJoin: 'round',
              }}
            />
          </>
        )}
        {failureOverlayActive && failureOverlay ? (
          <>
            <Polyline
              positions={failureLinePositions}
              pathOptions={{
                className: 'routeFailurePath',
                color: 'rgba(239, 68, 68, 0.96)',
                weight: 4,
                opacity: 0.95,
                lineCap: 'round',
                lineJoin: 'round',
                dashArray: '10 8',
              }}
            />
            {failureMidpoint ? (
              <CircleMarker
                center={[failureMidpoint.lat, failureMidpoint.lon]}
                radius={7}
                pathOptions={{
                  className: 'routeFailureBadge',
                  color: 'rgba(255, 255, 255, 0.95)',
                  weight: 1.5,
                  fillColor: 'rgba(239, 68, 68, 0.96)',
                  fillOpacity: 0.98,
                }}
              >
                <Tooltip
                  direction="top"
                  offset={[0, -6]}
                  opacity={1}
                  className="routeFailureTooltip"
                  permanent={false}
                >
                  <div className="routeFailureTooltip__title">Route compute failed</div>
                  <div className="routeFailureTooltip__row">reason_code={failureOverlay.reason_code}</div>
                  <div className="routeFailureTooltip__row">
                    {failureOverlay.stage
                      ? `stage=${failureOverlay.stage}${
                          failureOverlay.stage_detail ? `; detail=${failureOverlay.stage_detail}` : ''
                        }`
                      : 'No route candidate selected'}
                  </div>
                  <div className="routeFailureTooltip__row">{failureOverlay.message}</div>
                </Tooltip>
              </CircleMarker>
            ) : null}
          </>
        ) : null}

        {segmentBuckets.map((bucket) => {
          const positions = bucket.coordinates.map(([lon, lat]) => [lat, lon] as [number, number]);
          if (positions.length < 2) return null;
          const isActive = activeSegmentId === bucket.id;
          return (
            <Polyline
              key={bucket.id}
              positions={positions}
              eventHandlers={{
                mouseover() {
                  setActiveSegmentId(bucket.id);
                  onTutorialAction?.('map.segment_tooltip_hover');
                },
                mouseout() {
                  setActiveSegmentId((prev) => (prev === bucket.id ? null : prev));
                },
              }}
              pathOptions={{
                className: `segmentOverlayPath ${isActive ? 'segmentOverlayPath--active' : ''}`,
                color: isActive ? 'rgba(245, 158, 11, 0.94)' : 'rgba(241, 245, 249, 0.48)',
                weight: isActive ? 6 : 4,
                opacity: isActive ? 0.95 : 0.75,
                lineCap: 'round',
                lineJoin: 'round',
              }}
            >
              <Tooltip
                sticky={true}
                direction="top"
                offset={[0, -2]}
                className="segmentTooltip"
                opacity={1}
              >
                <div className="segmentTooltip__card">
                  <div className="segmentTooltip__title">
                    {overlayLabels?.segmentLabel ?? 'Segment'} {bucket.label}
                  </div>
                  <div className="segmentTooltip__row">Distance: {bucket.distance_km.toFixed(3)} km</div>
                  <div className="segmentTooltip__row">ETA: {bucket.duration_s.toFixed(1)} s</div>
                  <div className="segmentTooltip__row">
                    Cost: {bucket.monetary_cost.toFixed(3)} GBP
                  </div>
                  <div className="segmentTooltip__row">CO2: {bucket.emissions_kg.toFixed(3)} kg</div>
                  <div className="segmentTooltip__row">
                    Incident delay: {bucket.incident_delay_s.toFixed(1)} s
                  </div>
                </div>
              </Tooltip>
            </Polyline>
          );
        })}

        {showPreviewConnector && !failureOverlayActive && previewDotSegments.map((segment) => (
          <Polyline
            key={segment.id}
            positions={[
              [segment.from[1], segment.from[0]],
              [segment.to[1], segment.to[0]],
            ]}
            interactive={false}
            pathOptions={{
              className: 'stopPreviewDot',
              color: segment.color,
              weight: 3,
              opacity: 0.9,
              lineCap: 'round',
              lineJoin: 'round',
            }}
          />
        ))}
        {stopOverlayPoints.map((point) => {
          const isDuty = point.kind === 'duty';
          if (!isDuty) {
            return (
              <CircleMarker
                key={point.id}
                center={[point.lat, point.lon]}
                radius={7}
                interactive={false}
                pathOptions={{
                  className: point.kind === 'origin' ? 'stopOverlayOrigin' : 'stopOverlayDestination',
                  color: point.kind === 'origin' ? 'rgba(124, 58, 237, 0.95)' : 'rgba(6, 182, 212, 0.95)',
                  weight: 2,
                  fillColor: 'rgba(15, 23, 42, 0.72)',
                  fillOpacity: 0.85,
                }}
              />
            );
          }

          return null;
        })}

        {incidentOverlayPoints.map((point) => {
          const palette = incidentPalette(point.event.event_type);
          const incidentLabel =
            overlayLabels?.incidentTypeLabels[point.event.event_type] ?? point.event.event_type;
          return (
            <CircleMarker
              key={point.id}
              center={[point.lat, point.lon]}
              radius={6}
              pathOptions={{
                className: `incidentDot incidentDot--${point.event.event_type}`,
                color: palette.stroke,
                weight: 2,
                fillColor: palette.fill,
                fillOpacity: 0.95,
              }}
            >
              <Popup className="incidentPopup" autoPan={true} autoPanPadding={[22, 22]}>
                <div className="overlayPopup__card">
                  <div className="overlayPopup__title">{incidentLabel}</div>
                  <div className="overlayPopup__row">Delay: {point.event.delay_s.toFixed(1)} s</div>
                  <div className="overlayPopup__row">
                    Start offset: {point.event.start_offset_s.toFixed(1)} s
                  </div>
                  <div className="overlayPopup__row">Segment: {point.event.segment_index + 1}</div>
                  <div className="overlayPopup__row">Source: {point.event.source}</div>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}

        {timeLapsePosition && (
          <CircleMarker
            center={[timeLapsePosition.lat, timeLapsePosition.lon]}
            radius={8}
            pathOptions={{
              color: 'rgba(255, 255, 255, 0.95)',
              weight: 2,
              fillColor: 'rgba(6, 182, 212, 0.95)',
              fillOpacity: 0.95,
            }}
          />
        )}
      </MapContainer>
    </div>
  );
}
