import type {
  DutyChainStop,
  IncidentEventType,
  LatLng,
  ManagedStop,
  PinDisplayNode,
  RouteOption,
  RouteSegmentBreakdownRow,
  SimulatedIncidentEvent,
} from './types';

export type StopOverlayPoint = {
  id: string;
  kind: 'origin' | 'destination' | 'duty';
  lat: number;
  lon: number;
  sequence: number;
  label?: string | null;
};

export type IncidentOverlayPoint = {
  id: string;
  lat: number;
  lon: number;
  progress01: number;
  event: SimulatedIncidentEvent;
};

export type SegmentBucket = {
  id: string;
  startSegmentIndex: number;
  endSegmentIndex: number;
  label: string;
  coordinates: [number, number][];
  distance_km: number;
  duration_s: number;
  monetary_cost: number;
  emissions_kg: number;
  incident_delay_s: number;
  avg_speed_kmh: number;
};

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function toFiniteNumber(value: unknown, fallback = 0): number {
  const num = Number(value);
  return Number.isFinite(num) ? num : fallback;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  if (value >= 1) return 1;
  return value;
}

function sameCoord(aLat: number, aLon: number, bLat: number, bLon: number): boolean {
  return Math.abs(aLat - bLat) <= 1e-6 && Math.abs(aLon - bLon) <= 1e-6;
}

function hasPoint(points: StopOverlayPoint[], lat: number, lon: number): boolean {
  return points.some((point) => sameCoord(point.lat, point.lon, lat, lon));
}

function normaliseIncidentType(value: unknown): IncidentEventType | null {
  if (value === 'dwell' || value === 'accident' || value === 'closure') return value;
  return null;
}

export function parseDutyStopsForOverlay(text: string): DutyChainStop[] {
  if (!text.trim()) return [];
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  const out: DutyChainStop[] = [];
  for (const line of lines) {
    const parts = line.split(',');
    if (parts.length < 2) continue;

    const lat = Number(parts[0].trim());
    const lon = Number(parts[1].trim());
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    if (lat < -90 || lat > 90 || lon < -180 || lon > 180) continue;

    const label = parts.slice(2).join(',').trim();
    out.push({
      lat,
      lon,
      ...(label ? { label } : {}),
    });

    if (out.length >= 200) break;
  }
  return out;
}

export function interpolatePointAlongRoute(
  coords: [number, number][],
  progress01: number,
): [number, number] | null {
  if (coords.length === 0) return null;
  if (coords.length === 1) return coords[0];

  const progress = clamp01(progress01);
  if (progress <= 0) return coords[0];
  if (progress >= 1) return coords[coords.length - 1];

  const edgeLengths: number[] = [];
  let total = 0;
  for (let idx = 1; idx < coords.length; idx += 1) {
    const [lonA, latA] = coords[idx - 1];
    const [lonB, latB] = coords[idx];
    const len = Math.hypot(lonB - lonA, latB - latA);
    edgeLengths.push(len);
    total += len;
  }

  if (total <= 0) return coords[0];
  const target = total * progress;
  let walked = 0;

  for (let idx = 0; idx < edgeLengths.length; idx += 1) {
    const edge = edgeLengths[idx];
    const next = walked + edge;
    if (target <= next || idx === edgeLengths.length - 1) {
      const [lonA, latA] = coords[idx];
      const [lonB, latB] = coords[idx + 1];
      const local = edge > 0 ? (target - walked) / edge : 0;
      return [lonA + (lonB - lonA) * local, latA + (latB - latA) * local];
    }
    walked = next;
  }

  return coords[coords.length - 1];
}

export function buildStopOverlayPoints(
  origin: LatLng | null,
  destination: LatLng | null,
  dutyStops: DutyChainStop[],
): StopOverlayPoint[] {
  const points: StopOverlayPoint[] = [];
  let dutySequence = 1;

  if (origin) {
    points.push({
      id: 'stop-origin',
      kind: 'origin',
      lat: origin.lat,
      lon: origin.lon,
      sequence: 0,
      label: 'Start',
    });
  }

  for (let idx = 0; idx < dutyStops.length; idx += 1) {
    const stop = dutyStops[idx];
    if (!isFiniteNumber(stop.lat) || !isFiniteNumber(stop.lon)) continue;
    if (stop.lat < -90 || stop.lat > 90 || stop.lon < -180 || stop.lon > 180) continue;
    if (hasPoint(points, stop.lat, stop.lon)) continue;

    points.push({
      id: `stop-duty-${idx + 1}`,
      kind: 'duty',
      lat: stop.lat,
      lon: stop.lon,
      sequence: dutySequence,
      label: stop.label ?? null,
    });
    dutySequence += 1;
  }

  if (destination && !hasPoint(points, destination.lat, destination.lon)) {
    points.push({
      id: 'stop-destination',
      kind: 'destination',
      lat: destination.lat,
      lon: destination.lon,
      sequence: 0,
      label: 'Destination',
    });
  }

  return points;
}

export function buildManagedPinNodes(
  origin: LatLng | null,
  destination: LatLng | null,
  stop: ManagedStop | null,
  labels: {
    origin?: string;
    destination?: string;
  } = {},
): PinDisplayNode[] {
  const out: PinDisplayNode[] = [];
  if (origin) {
    out.push({
      id: 'origin',
      kind: 'origin',
      lat: origin.lat,
      lon: origin.lon,
      label: labels.origin?.trim() || 'Start',
      order: 1,
    });
  }
  if (stop) {
    out.push({
      id: 'stop-1',
      kind: 'stop',
      lat: stop.lat,
      lon: stop.lon,
      label: stop.label?.trim() || 'Stop #1',
      order: 2,
    });
  }
  if (destination) {
    out.push({
      id: 'destination',
      kind: 'destination',
      lat: destination.lat,
      lon: destination.lon,
      label: labels.destination?.trim() || 'Destination',
      order: stop ? 3 : 2,
    });
  }
  return out;
}

export function buildIncidentOverlayPoints(route: RouteOption | null): IncidentOverlayPoint[] {
  if (!route) return [];
  const coords = route.geometry?.coordinates ?? [];
  if (!Array.isArray(coords) || coords.length < 2) return [];

  const events = Array.isArray(route.incident_events) ? route.incident_events : [];
  if (!events.length) return [];

  const durationTotalS = toFiniteNumber(route.metrics?.duration_s, 0);
  const segmentCountFromBreakdown = Array.isArray(route.segment_breakdown)
    ? route.segment_breakdown.length
    : 0;
  const segmentCount = Math.max(1, segmentCountFromBreakdown);

  const out: IncidentOverlayPoint[] = [];
  for (let idx = 0; idx < events.length; idx += 1) {
    const raw = events[idx];
    const eventType = normaliseIncidentType(raw?.event_type);
    if (!eventType) continue;
    const event: SimulatedIncidentEvent = {
      event_id: String(raw.event_id || `incident-${idx + 1}`),
      event_type: eventType,
      segment_index: Math.max(0, Math.floor(toFiniteNumber(raw.segment_index, 0))),
      start_offset_s: Math.max(0, toFiniteNumber(raw.start_offset_s, 0)),
      delay_s: Math.max(0, toFiniteNumber(raw.delay_s, 0)),
      source: 'synthetic',
    };

    let progress =
      durationTotalS > 0 ? clamp01(toFiniteNumber(event.start_offset_s, 0) / durationTotalS) : NaN;
    if (!Number.isFinite(progress)) {
      progress = clamp01(toFiniteNumber(event.segment_index, 0) / segmentCount);
    }
    const point = interpolatePointAlongRoute(coords, progress);
    if (!point) continue;
    out.push({
      id: event.event_id,
      lon: point[0],
      lat: point[1],
      progress01: progress,
      event,
    });
  }

  out.sort((a, b) => a.progress01 - b.progress01);
  return out;
}

function normaliseSegmentRows(route: RouteOption): RouteSegmentBreakdownRow[] {
  const rows = Array.isArray(route.segment_breakdown) ? route.segment_breakdown : [];
  return rows
    .map((row, idx) => {
      const segIdx = Math.max(0, Math.floor(toFiniteNumber(row.segment_index, idx)));
      return {
        segment_index: segIdx,
        distance_km: Math.max(0, toFiniteNumber(row.distance_km, 0)),
        duration_s: Math.max(0, toFiniteNumber(row.duration_s, 0)),
        incident_delay_s: Math.max(0, toFiniteNumber(row.incident_delay_s, 0)),
        avg_speed_kmh: Math.max(0, toFiniteNumber(row.avg_speed_kmh, 0)),
        emissions_kg: Math.max(0, toFiniteNumber(row.emissions_kg, 0)),
        monetary_cost: Math.max(0, toFiniteNumber(row.monetary_cost, 0)),
      };
    })
    .sort((a, b) => a.segment_index - b.segment_index);
}

export function buildSegmentBuckets(route: RouteOption | null, maxBuckets = 120): SegmentBucket[] {
  if (!route) return [];
  const coords = route.geometry?.coordinates ?? [];
  if (!Array.isArray(coords) || coords.length < 2) return [];

  const segments = normaliseSegmentRows(route);
  if (!segments.length) return [];

  const coordEdgeCount = coords.length - 1;
  const clampedMaxBuckets = Math.max(1, Math.min(2000, Math.floor(maxBuckets)));
  const bucketCount = Math.min(clampedMaxBuckets, segments.length);
  const bucketSize = Math.max(1, Math.ceil(segments.length / bucketCount));

  const buckets: SegmentBucket[] = [];
  for (let start = 0; start < segments.length; start += bucketSize) {
    const end = Math.min(segments.length - 1, start + bucketSize - 1);
    const rows = segments.slice(start, end + 1);

    let distanceKm = 0;
    let durationS = 0;
    let money = 0;
    let co2 = 0;
    let incidentDelayS = 0;
    for (const row of rows) {
      distanceKm += row.distance_km;
      durationS += row.duration_s;
      money += row.monetary_cost;
      co2 += row.emissions_kg;
      incidentDelayS += row.incident_delay_s ?? 0;
    }

    let startCoordIdx = Math.floor((start / segments.length) * coordEdgeCount);
    let endCoordIdx = Math.ceil(((end + 1) / segments.length) * coordEdgeCount);
    startCoordIdx = Math.max(0, Math.min(coordEdgeCount, startCoordIdx));
    endCoordIdx = Math.max(0, Math.min(coordEdgeCount, endCoordIdx));
    if (endCoordIdx <= startCoordIdx) {
      endCoordIdx = Math.min(coordEdgeCount, startCoordIdx + 1);
    }
    const line = coords.slice(startCoordIdx, endCoordIdx + 1);
    if (line.length < 2) continue;

    const label = start === end ? `S${start + 1}` : `S${start + 1}-S${end + 1}`;
    const avgSpeedKmh = durationS > 0 ? distanceKm / (durationS / 3600.0) : 0;
    buckets.push({
      id: `segment-bucket-${start + 1}-${end + 1}`,
      startSegmentIndex: start,
      endSegmentIndex: end,
      label,
      coordinates: line,
      distance_km: distanceKm,
      duration_s: durationS,
      monetary_cost: money,
      emissions_kg: co2,
      incident_delay_s: incidentDelayS,
      avg_speed_kmh: avgSpeedKmh,
    });
  }

  return buckets;
}
