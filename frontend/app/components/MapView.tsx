'use client';
// frontend/app/components/MapView.tsx

import L, {
  type LatLngBoundsExpression,
  type LatLngExpression,
  type LeafletMouseEvent,
} from 'leaflet';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  MapContainer,
  Marker,
  Polyline,
  Popup,
  TileLayer,
  useMap,
  useMapEvents,
} from 'react-leaflet';

import type { LatLng, RouteOption } from '../lib/types';

export type MarkerKind = 'origin' | 'destination';
const MAX_POLYLINE_POINTS = 1000;

type Props = {
  origin: LatLng | null;
  destination: LatLng | null;

  selectedMarker: MarkerKind | null;

  route: RouteOption | null;

  onMapClick: (lat: number, lon: number) => void;
  onSelectMarker: (kind: MarkerKind | null) => void;
  onMoveMarker: (kind: MarkerKind, lat: number, lon: number) => void;
  onRemoveMarker: (kind: MarkerKind) => void;
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
  const label = kind === 'origin' ? 'S' : 'E';
  const baseClass = kind === 'origin' ? 'pin-origin' : 'pin-destination';
  const selectedClass = selected ? 'pin--selected' : '';

  return L.divIcon({
    className: `pin ${baseClass} ${selectedClass}`.trim(),
    html: `<div class="pin__inner">${label}</div>`,
    iconSize: [34, 44],
    iconAnchor: [17, 44],
    popupAnchor: [0, -42],
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

function Recenter({ center }: { center: LatLngExpression }) {
  const map = useMap();

  // IMPORTANT:
  // Using a stable callback avoids Prettier ever producing the invalid:
  //   };, [map, center]);
  // because we no longer write the effect callback inline.
  const doRecenter = useCallback(() => {
    map.flyTo(center, map.getZoom(), { animate: true, duration: 0.55 });
  }, [map, center]);

  useEffect(doRecenter, [doRecenter]);

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

function TrashIcon() {
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
      <path d="M3 6h18" />
      <path d="M8 6V4h8v2" />
      <path d="M6 6l1 16h10l1-16" />
      <path d="M10 11v6" />
      <path d="M14 11v6" />
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

export default function MapView({
  origin,
  destination,
  selectedMarker,
  route,
  onMapClick,
  onSelectMarker,
  onMoveMarker,
  onRemoveMarker,
  onSwapMarkers,
}: Props) {
  const originRef = useRef<L.Marker>(null);
  const destRef = useRef<L.Marker>(null);

  const [copied, setCopied] = useState<MarkerKind | null>(null);

  useEffect(() => {
    if (!copied) return;
    const t = window.setTimeout(() => setCopied(null), 900);
    return () => window.clearTimeout(t);
  }, [copied]);

  const originIcon = useMemo(
    () => makePinIcon('origin', selectedMarker === 'origin'),
    [selectedMarker],
  );
  const destIcon = useMemo(
    () => makePinIcon('destination', selectedMarker === 'destination'),
    [selectedMarker],
  );

  useEffect(() => {
    try {
      if (originRef.current) {
        if (selectedMarker === 'origin') originRef.current.openPopup();
        else originRef.current.closePopup();
      }
      if (destRef.current) {
        if (selectedMarker === 'destination') destRef.current.openPopup();
        else destRef.current.closePopup();
      }
    } catch {
      // no-op
    }
  }, [selectedMarker]);

  const polylinePositions: LatLngExpression[] = useMemo(() => {
    const coords = route?.geometry?.coordinates ?? [];
    const slim = downsamplePolyline(coords);
    return slim.map(([lon, lat]) => [lat, lon] as [number, number]);
  }, [route]);

  // Default center: Birmingham-ish (West Midlands)
  const center: LatLngExpression = useMemo(() => {
    return origin
      ? ([origin.lat, origin.lon] as [number, number])
      : ([52.4862, -1.8904] as [number, number]);
  }, [origin]);

  const handleCopyCoords = useCallback(async (kind: MarkerKind, lat: number, lon: number) => {
    const text = `${fmtCoord(lat)}, ${fmtCoord(lon)}`;
    const ok = await copyToClipboard(text);
    if (ok) setCopied(kind);
  }, []);

  return (
    <div className="mapPane">
      <MapContainer
        key={MAP_HOT_RELOAD_KEY}
        center={center}
        zoom={11}
        minZoom={5}
        maxZoom={18}
        maxBounds={UK_BOUNDS}
        maxBoundsViscosity={1.0}
        style={{ height: '100%', width: '100%' }}
      >
        <Recenter center={center} />

        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          subdomains={['a', 'b', 'c', 'd']}
        />

        <ClickHandler onMapClick={onMapClick} />

        {origin && (
          <Marker
            ref={originRef}
            position={[origin.lat, origin.lon]}
            icon={originIcon}
            draggable={true}
            riseOnHover={true}
            eventHandlers={{
              click(e) {
                e.originalEvent?.stopPropagation();
                onSelectMarker(selectedMarker === 'origin' ? null : 'origin');
              },
              dragend(e) {
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                onMoveMarker('origin', pos.lat, pos.lng);
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
                  <span className="markerPopup__pill markerPopup__pill--origin">Start</span>

                  <div className="markerPopup__actions">
                    {onSwapMarkers && destination && (
                      <button
                        type="button"
                        className="markerPopup__iconBtn"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSwapMarkers();
                        }}
                        aria-label="Swap start and destination"
                        title="Swap start and destination"
                      >
                        <SwapIcon />
                      </button>
                    )}

                    <button
                      type="button"
                      className="markerPopup__iconBtn markerPopup__iconBtn--danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemoveMarker('origin');
                        onSelectMarker(null);
                      }}
                      aria-label="Remove start"
                      title="Remove start"
                    >
                      <TrashIcon />
                    </button>

                    <button
                      type="button"
                      className="markerPopup__iconBtn"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectMarker(null);
                      }}
                      aria-label="Close"
                      title="Close"
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
                  >
                    <span className="markerPopup__coordsText">
                      {fmtCoord(origin.lat)}, {fmtCoord(origin.lon)}
                    </span>
                    <span className="markerPopup__coordsIcon">
                      <CopyIcon />
                    </span>
                  </button>

                  {copied === 'origin' && <span className="markerPopup__toast">Copied</span>}
                </div>
              </div>
            </Popup>
          </Marker>
        )}

        {destination && (
          <Marker
            ref={destRef}
            position={[destination.lat, destination.lon]}
            icon={destIcon}
            draggable={true}
            riseOnHover={true}
            eventHandlers={{
              click(e) {
                e.originalEvent?.stopPropagation();
                onSelectMarker(selectedMarker === 'destination' ? null : 'destination');
              },
              dragend(e) {
                const marker = e.target as L.Marker;
                const pos = marker.getLatLng();
                onMoveMarker('destination', pos.lat, pos.lng);
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
                    Destination
                  </span>

                  <div className="markerPopup__actions">
                    {onSwapMarkers && origin && (
                      <button
                        type="button"
                        className="markerPopup__iconBtn"
                        onClick={(e) => {
                          e.stopPropagation();
                          onSwapMarkers();
                        }}
                        aria-label="Swap start and destination"
                        title="Swap start and destination"
                      >
                        <SwapIcon />
                      </button>
                    )}

                    <button
                      type="button"
                      className="markerPopup__iconBtn markerPopup__iconBtn--danger"
                      onClick={(e) => {
                        e.stopPropagation();
                        onRemoveMarker('destination');
                        onSelectMarker(null);
                      }}
                      aria-label="Remove destination"
                      title="Remove destination"
                    >
                      <TrashIcon />
                    </button>

                    <button
                      type="button"
                      className="markerPopup__iconBtn"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectMarker(null);
                      }}
                      aria-label="Close"
                      title="Close"
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
                  >
                    <span className="markerPopup__coordsText">
                      {fmtCoord(destination.lat)}, {fmtCoord(destination.lon)}
                    </span>
                    <span className="markerPopup__coordsIcon">
                      <CopyIcon />
                    </span>
                  </button>

                  {copied === 'destination' && <span className="markerPopup__toast">Copied</span>}
                </div>
              </div>
            </Popup>
          </Marker>
        )}

        {polylinePositions.length > 0 && (
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
      </MapContainer>
    </div>
  );
}
