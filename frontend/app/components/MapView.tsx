'use client';

import L, {
  type LatLngBoundsExpression,
  type LatLngExpression,
  type LeafletMouseEvent,
} from 'leaflet';
import { useEffect, useMemo } from 'react';
import { MapContainer, Marker, Polyline, TileLayer, useMap, useMapEvents } from 'react-leaflet';

import type { LatLng, RouteOption } from '../lib/types';

type Props = {
  origin: LatLng | null;
  destination: LatLng | null;
  route: RouteOption | null;
  onMapClick: (lat: number, lon: number) => void;
};

// Rough UK bounds (keeps the demo focused on UK routing).
const UK_BOUNDS: LatLngBoundsExpression = [
  [49.5, -8.7],
  [61.1, 2.1],
];

const ORIGIN_ICON = L.divIcon({
  className: 'pin pin-origin',
  html: '<div class="pin__inner">O</div>',
  iconSize: [34, 44],
  iconAnchor: [17, 44],
});

const DEST_ICON = L.divIcon({
  className: 'pin pin-destination',
  html: '<div class="pin__inner">D</div>',
  iconSize: [34, 44],
  iconAnchor: [17, 44],
});

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
  useEffect(() => {
    // Smooth camera motion feels much more modern than a hard jump.
    map.flyTo(center, map.getZoom(), { animate: true, duration: 0.55 });
  }, [map, center]);
  return null;
}

export default function MapView({ origin, destination, route, onMapClick }: Props) {
  const polylinePositions: LatLngExpression[] = useMemo(() => {
    return route?.geometry?.coordinates?.map(([lon, lat]) => [lat, lon] as [number, number]) ?? [];
  }, [route]);

  // Default center: Birmingham-ish (West Midlands)
  const center: LatLngExpression = useMemo(() => {
    return origin
      ? ([origin.lat, origin.lon] as [number, number])
      : ([52.4862, -1.8904] as [number, number]);
  }, [origin]);

  return (
    <div className="mapPane">
      <MapContainer
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

        {origin && <Marker position={[origin.lat, origin.lon]} icon={ORIGIN_ICON} />}
        {destination && <Marker position={[destination.lat, destination.lon]} icon={DEST_ICON} />}

        {polylinePositions.length > 0 && (
          <>
            {/* Glow underlay */}
            <Polyline
              positions={polylinePositions}
              pathOptions={{
                className: 'routeGlow',
                color: 'rgba(6, 182, 212, 0.35)',
                weight: 12,
                opacity: 0.9,
              }}
            />
            {/* Animated main stroke */}
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
