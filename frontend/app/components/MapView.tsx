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
  managedStop?: ManagedStop | null;
  originLabel?: string;
  destinationLabel?: string;

  selectedPinId?: PinSelectionId | null;
  focusPinRequest?: PinFocusRequest | null;
  fitAllRequestNonce?: number;

  route: RouteOption | null;
  timeLapsePosition?: LatLng | null;
  dutyStops?: DutyChainStop[];
  showStopOverlay?: boolean;
  showIncidentOverlay?: boolean;
  showSegmentTooltips?: boolean;
  overlayLabels?: {
    stopLabel: string;
    segmentLabel: string;
    incidentTypeLabels: Record<'dwell' | 'accident' | 'closure', string>;
  };
  onTutorialAction?: (actionId: string) => void;
  onTutorialTargetState?: (state: { hasSegmentTooltipPath: boolean; hasIncidentMarkers: boolean }) => void;
  tutorialMapLocked?: boolean;
  tutorialViewportLocked?: boolean;
  tutorialGuideTarget?: TutorialGuideTarget | null;
  tutorialGuideVisible?: boolean;

  onMapClick: (lat: number, lon: number) => void;
  onSelectPinId?: (id: PinSelectionId | null) => void;
  onMoveMarker: (kind: MarkerKind, lat: number, lon: number) => void;
  onMoveStop?: (lat: number, lon: number) => void;
  onAddStopFromPin?: () => void;
  onRenameStop?: (name: string) => void;
  onDeleteStop?: () => void;
  onFocusPin?: (id: PinSelectionId) => void;
  onSwapMarkers?: () => void;
};

// Rough UK bounds (keeps the demo focused on UK routing).
const UK_BOUNDS: LatLngBoundsExpression = [
  [49.5, -8.7],
  [61.1, 2.1],
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
  useMapEvents({
    click(e: LeafletMouseEvent) {
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

    const nextZoom = Math.min(14, Math.max(map.getZoom() + 1, 11));
    map.flyTo([target.lat, target.lon], nextZoom, {
      animate: true,
      duration: 0.45,
    });

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
}: {
  nonce: number;
  origin: LatLng | null;
  destination: LatLng | null;
  managedStop: ManagedStop | null;
  route: RouteOption | null;
}) {
  const map = useMap();

  useEffect(() => {
    if (nonce <= 0) return;
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

    if (points.length >= 2) {
      map.fitBounds(points as LatLngBoundsExpression, {
        animate: true,
        duration: 0.45,
        padding: [36, 36],
      });
      return;
    }
    if (points.length === 1) {
      const [lat, lon] = points[0] as [number, number];
      map.flyTo([lat, lon], Math.min(map.getZoom(), 11), {
        animate: true,
        duration: 0.35,
      });
    }
  }, [nonce, origin, destination, managedStop, route?.id, map]);

  return null;
}

function TutorialGuidePanHandler({
  guideTarget,
  visible,
}: {
  guideTarget: TutorialGuideTarget | null | undefined;
  visible: boolean;
}) {
  const map = useMap();

  useEffect(() => {
    if (!visible || !guideTarget) return;
    map.flyTo([guideTarget.lat, guideTarget.lon], guideTarget.zoom, {
      animate: true,
      duration: 0.55,
    });
  }, [guideTarget?.pan_nonce, guideTarget?.stage, visible, map, guideTarget]);

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
  managedStop = null,
  originLabel = 'Start',
  destinationLabel = 'End',
  selectedPinId = null,
  focusPinRequest = null,
  fitAllRequestNonce = 0,
  route,
  timeLapsePosition,
  dutyStops = [],
  showStopOverlay = true,
  showIncidentOverlay = true,
  showSegmentTooltips = true,
  overlayLabels,
  onTutorialAction,
  onTutorialTargetState,
  tutorialMapLocked = false,
  tutorialViewportLocked = false,
  tutorialGuideTarget = null,
  tutorialGuideVisible = false,
  onMapClick,
  onSelectPinId,
  onMoveMarker,
  onMoveStop,
  onAddStopFromPin,
  onRenameStop,
  onDeleteStop,
  onFocusPin,
  onSwapMarkers,
}: Props) {
  const originRef = useRef<L.Marker>(null);
  const destRef = useRef<L.Marker>(null);
  const stopRef = useRef<L.Marker>(null);

  const [copied, setCopied] = useState<MarkerKind | 'stop' | null>(null);
  const [activeSegmentId, setActiveSegmentId] = useState<string | null>(null);
  const [stopDraftLabel, setStopDraftLabel] = useState(managedStop?.label ?? 'Stop #1');
  const [draggingPinId, setDraggingPinId] = useState<PinSelectionId | null>(null);
  const suppressMapClickUntilRef = useRef(0);
  const suppressMarkerClickUntilRef = useRef(0);
  const [dragPreview, setDragPreview] = useState<{
    origin: LatLng | null;
    destination: LatLng | null;
    stop: LatLng | null;
  }>({
    origin: null,
    destination: null,
    stop: null,
  });

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
    const until = Date.now() + 820;
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

  const effectiveOrigin = dragPreview.origin ?? origin;
  const effectiveDestination = dragPreview.destination ?? destination;
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

  const stopOverlayPoints = useMemo(() => {
    if (!showStopOverlay) return [];
    return buildStopOverlayPoints(effectiveOrigin, effectiveDestination, effectiveDutyStops);
  }, [showStopOverlay, effectiveOrigin, effectiveDestination, effectiveDutyStops]);
  const previewDotSegments = useMemo(
    () => buildPreviewRouteSegmentsFromNodes(previewNodes),
    [previewNodes],
  );

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

  // Default center: Birmingham-ish (West Midlands)
  const center: LatLngExpression = useMemo(() => {
    return effectiveOrigin
      ? ([effectiveOrigin.lat, effectiveOrigin.lon] as [number, number])
      : ([52.4862, -1.8904] as [number, number]);
  }, [effectiveOrigin]);

  const suppressRoutePath = draggingPinId !== null;

  const handleCopyCoords = useCallback(async (kind: MarkerKind | 'stop', lat: number, lon: number) => {
    const text = `${fmtCoord(lat)}, ${fmtCoord(lon)}`;
    const ok = await copyToClipboard(text);
    if (ok) {
      setCopied(kind);
      onTutorialAction?.('map.popup_copy');
    }
  }, [onTutorialAction]);

  const handleMapClickFromMap = useCallback(
    (lat: number, lon: number) => {
      if (tutorialMapLocked) return;
      if (draggingPinId !== null) return;
      if (Date.now() < suppressMapClickUntilRef.current) return;
      onMapClick(lat, lon);
    },
    [draggingPinId, onMapClick, tutorialMapLocked],
  );

  return (
    <div
      className={`mapPane ${tutorialMapLocked ? 'mapPane--tutorialLocked' : ''} ${tutorialGuideVisible ? 'mapPane--guide' : ''}`.trim()}
      data-segment-tooltips={showSegmentTooltips ? 'on' : 'off'}
      data-tutorial-id="map.interactive"
    >
      <MapContainer
        key={MAP_HOT_RELOAD_KEY}
        center={center}
        zoom={11}
        minZoom={5}
        maxZoom={18}
        dragging={!tutorialMapLocked && !tutorialViewportLocked}
        scrollWheelZoom={!tutorialMapLocked && !tutorialViewportLocked}
        doubleClickZoom={!tutorialMapLocked && !tutorialViewportLocked}
        boxZoom={!tutorialMapLocked && !tutorialViewportLocked}
        keyboard={!tutorialMapLocked && !tutorialViewportLocked}
        touchZoom={!tutorialMapLocked && !tutorialViewportLocked}
        maxBounds={UK_BOUNDS}
        maxBoundsViscosity={1.0}
        style={{ height: '100%', width: '100%' }}
      >
        <FocusPinRequestHandler
          request={focusPinRequest}
          origin={origin}
          destination={destination}
          managedStop={managedStop}
          originRef={originRef}
          destRef={destRef}
          stopRef={stopRef}
        />
        <FitAllRequestHandler
          nonce={fitAllRequestNonce}
          origin={origin}
          destination={destination}
          managedStop={managedStop}
          route={route}
        />
        <TutorialGuidePanHandler guideTarget={tutorialGuideTarget} visible={tutorialGuideVisible} />

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

        {origin && (
          <Marker
            ref={originRef}
            position={[effectiveOrigin?.lat ?? origin.lat, effectiveOrigin?.lon ?? origin.lon]}
            icon={originIcon}
            draggable={true}
            riseOnHover={true}
            eventHandlers={{
              click(e) {
                if (tutorialMapLocked) return;
                e.originalEvent?.stopPropagation();
                if (Date.now() < suppressMarkerClickUntilRef.current) return;
                onSelectPinId?.(selectedPinId === 'origin' ? null : 'origin');
                onFocusPin?.('origin');
                onTutorialAction?.('map.click_origin_marker');
              },
              dragend(e) {
                if (tutorialMapLocked) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, origin: { lat: pos.lat, lon: pos.lng } }));
                suppressInteractionsBriefly();
                onMoveMarker('origin', pos.lat, pos.lng);
                onTutorialAction?.('map.drag_origin_marker');
              },
              dragstart() {
                if (tutorialMapLocked) return;
                suppressInteractionsBriefly();
                try {
                  originRef.current?.closePopup();
                } catch {
                  // no-op
                }
                setDraggingPinId('origin');
              },
              drag(e) {
                if (tutorialMapLocked) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, origin: { lat: pos.lat, lon: pos.lng } }));
              },
            }}
          >
            <Popup
              className="markerPopup"
              closeButton={false}
              autoPan={true}
              autoPanPadding={[22, 22]}
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
                        onClick={(e) => {
                          e.stopPropagation();
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
                        onClick={(e) => {
                          e.stopPropagation();
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

                <div className="markerPopup__coordsRow">
                  <button
                    type="button"
                    className="markerPopup__coordsBtn"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCopyCoords('origin', origin.lat, origin.lon);
                    }}
                    aria-label="Copy coordinates"
                    title="Copy coordinates"
                    data-tutorial-action="map.popup_copy"
                  >
                    <span className="markerPopup__coordsText">
                      {fmtCoord(effectiveOrigin?.lat ?? origin.lat)}, {fmtCoord(effectiveOrigin?.lon ?? origin.lon)}
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
          </Marker>
        )}

        {destination && (
          <Marker
            ref={destRef}
            position={[effectiveDestination?.lat ?? destination.lat, effectiveDestination?.lon ?? destination.lon]}
            icon={destIcon}
            draggable={true}
            riseOnHover={true}
            eventHandlers={{
              click(e) {
                if (tutorialMapLocked) return;
                e.originalEvent?.stopPropagation();
                if (Date.now() < suppressMarkerClickUntilRef.current) return;
                onSelectPinId?.(selectedPinId === 'destination' ? null : 'destination');
                onFocusPin?.('destination');
                onTutorialAction?.('map.click_destination_marker');
              },
              dragend(e) {
                if (tutorialMapLocked) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, destination: { lat: pos.lat, lon: pos.lng } }));
                suppressInteractionsBriefly();
                onMoveMarker('destination', pos.lat, pos.lng);
                onTutorialAction?.('map.drag_destination_marker');
              },
              dragstart() {
                if (tutorialMapLocked) return;
                suppressInteractionsBriefly();
                try {
                  destRef.current?.closePopup();
                } catch {
                  // no-op
                }
                setDraggingPinId('destination');
              },
              drag(e) {
                if (tutorialMapLocked) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, destination: { lat: pos.lat, lon: pos.lng } }));
              },
            }}
          >
            <Popup
              className="markerPopup"
              closeButton={false}
              autoPan={true}
              autoPanPadding={[22, 22]}
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
                        onClick={(e) => {
                          e.stopPropagation();
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
                        onClick={(e) => {
                          e.stopPropagation();
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

                <div className="markerPopup__coordsRow">
                  <button
                    type="button"
                    className="markerPopup__coordsBtn"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCopyCoords('destination', destination.lat, destination.lon);
                    }}
                    aria-label="Copy coordinates"
                    title="Copy coordinates"
                    data-tutorial-action="map.popup_copy"
                  >
                    <span className="markerPopup__coordsText">
                      {fmtCoord(effectiveDestination?.lat ?? destination.lat)}, {fmtCoord(effectiveDestination?.lon ?? destination.lon)}
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
          </Marker>
        )}

        {showStopOverlay && managedStop ? (
          <Marker
            ref={stopRef}
            position={[effectiveStop?.lat ?? managedStop.lat, effectiveStop?.lon ?? managedStop.lon]}
            icon={stopIcon}
            draggable={true}
            riseOnHover={true}
            eventHandlers={{
              click(e) {
                if (tutorialMapLocked) return;
                e.originalEvent?.stopPropagation();
                if (Date.now() < suppressMarkerClickUntilRef.current) return;
                onSelectPinId?.(selectedPinId === 'stop-1' ? null : 'stop-1');
                onFocusPin?.('stop-1');
                onTutorialAction?.('map.click_stop_marker');
              },
              dragend(e) {
                if (tutorialMapLocked) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, stop: { lat: pos.lat, lon: pos.lng } }));
                suppressInteractionsBriefly();
                onMoveStop?.(pos.lat, pos.lng);
                onTutorialAction?.('map.drag_stop_marker');
              },
              dragstart() {
                if (tutorialMapLocked) return;
                suppressInteractionsBriefly();
                try {
                  stopRef.current?.closePopup();
                } catch {
                  // no-op
                }
                setDraggingPinId('stop-1');
              },
              drag(e) {
                if (tutorialMapLocked) return;
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                setDragPreview((prev) => ({ ...prev, stop: { lat: pos.lat, lon: pos.lng } }));
              },
            }}
          >
            <Popup className="stopOverlayPopup" autoPan={true} autoPanPadding={[22, 22]} closeButton={false}>
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
            <Tooltip permanent={true} direction="top" offset={[0, -16]} className="stopNameTooltip">
              {stopDisplayName}
            </Tooltip>
          </Marker>
        ) : null}

        {!suppressRoutePath && polylinePositions.length > 0 && (
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

        {previewDotSegments.map((segment) => (
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
