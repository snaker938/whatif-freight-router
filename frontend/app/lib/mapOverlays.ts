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

export type PreviewRouteNode = {
  id: string;
  role: 'start' | 'stop' | 'end';
  lat: number;
  lon: number;
  color: string;
};

export type PreviewDotSegment = {
  id: string;
  legIndex: number;
  segmentIndex: number;
  fromNodeId: string;
  toNodeId: string;
  from: [number, number];
  to: [number, number];
  color: string;
};

const PREVIEW_START_COLOR = '#7C3AED';
const PREVIEW_END_COLOR = '#06B6D4';
const EARTH_RADIUS_M = 6_371_000;

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

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min;
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

function toRad(value: number): number {
  return (value * Math.PI) / 180;
}

function toDeg(value: number): number {
  return (value * 180) / Math.PI;
}

function wrapAngle(angle: number): number {
  let out = angle;
  while (out <= -Math.PI) out += 2 * Math.PI;
  while (out > Math.PI) out -= 2 * Math.PI;
  return out;
}

function projectLonLat(lon: number, lat: number, refLatRad: number): [number, number] {
  const lonRad = toRad(lon);
  const latRad = toRad(lat);
  const x = lonRad * EARTH_RADIUS_M * Math.cos(refLatRad);
  const y = latRad * EARTH_RADIUS_M;
  return [x, y];
}

function unprojectXY(x: number, y: number, refLatRad: number): [number, number] {
  const lonRad = x / (EARTH_RADIUS_M * Math.cos(refLatRad));
  const latRad = y / EARTH_RADIUS_M;
  return [toDeg(lonRad), toDeg(latRad)];
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

function hexToRgb(hex: string): [number, number, number] {
  const raw = hex.trim().replace('#', '');
  if (raw.length !== 6) return [255, 255, 255];

  const r = Number.parseInt(raw.slice(0, 2), 16);
  const g = Number.parseInt(raw.slice(2, 4), 16);
  const b = Number.parseInt(raw.slice(4, 6), 16);
  if (!Number.isFinite(r) || !Number.isFinite(g) || !Number.isFinite(b)) {
    return [255, 255, 255];
  }
  return [r, g, b];
}

function interpolateHexColor(startHex: string, endHex: string, t: number): string {
  const [sr, sg, sb] = hexToRgb(startHex);
  const [er, eg, eb] = hexToRgb(endHex);
  const clamped = clamp01(t);
  const r = Math.round(sr + (er - sr) * clamped);
  const g = Math.round(sg + (eg - sg) * clamped);
  const b = Math.round(sb + (eb - sb) * clamped);
  const toHex = (value: number) => value.toString(16).padStart(2, '0');
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function hasRawPreviewNode(
  nodes: Array<Omit<PreviewRouteNode, 'color'>>,
  lat: number,
  lon: number,
): boolean {
  return nodes.some((node) => sameCoord(node.lat, node.lon, lat, lon));
}

export function buildPreviewRouteNodes(
  origin: LatLng | null,
  destination: LatLng | null,
  dutyStops: DutyChainStop[],
): PreviewRouteNode[] {
  const rawNodes: Array<Omit<PreviewRouteNode, 'color'>> = [];

  if (origin) {
    rawNodes.push({
      id: 'origin',
      role: 'start',
      lat: origin.lat,
      lon: origin.lon,
    });
  }

  for (let idx = 0; idx < dutyStops.length; idx += 1) {
    const stop = dutyStops[idx];
    if (!isFiniteNumber(stop.lat) || !isFiniteNumber(stop.lon)) continue;
    if (stop.lat < -90 || stop.lat > 90 || stop.lon < -180 || stop.lon > 180) continue;
    if (hasRawPreviewNode(rawNodes, stop.lat, stop.lon)) continue;

    rawNodes.push({
      id: `stop-${idx + 1}`,
      role: 'stop',
      lat: stop.lat,
      lon: stop.lon,
    });
  }

  if (destination && !hasRawPreviewNode(rawNodes, destination.lat, destination.lon)) {
    rawNodes.push({
      id: 'destination',
      role: 'end',
      lat: destination.lat,
      lon: destination.lon,
    });
  }

  if (rawNodes.length === 0) return [];
  const denom = Math.max(1, rawNodes.length - 1);
  return rawNodes.map((node, idx) => ({
    ...node,
    color: interpolateHexColor(PREVIEW_START_COLOR, PREVIEW_END_COLOR, idx / denom),
  }));
}

function buildRoadLikeLegPoints(
  start: [number, number],
  end: [number, number],
  legIndex: number,
): [number, number][] {
  const [startLon, startLat] = start;
  const [endLon, endLat] = end;
  const refLatRad = toRad((startLat + endLat) * 0.5);
  const [startX, startY] = projectLonLat(startLon, startLat, refLatRad);
  const [endX, endY] = projectLonLat(endLon, endLat, refLatRad);
  const dx = endX - startX;
  const dy = endY - startY;
  const totalDistanceM = Math.hypot(dx, dy);
  if (totalDistanceM <= 1) {
    return [start, end];
  }

  const targetSteps = Math.round(clamp(totalDistanceM / 1200, 18, 650));
  const stepM = Math.max(450, totalDistanceM / targetSteps);
  const quantStepRad = Math.PI / 12; // 15deg headings to mimic road-like legs
  const maxTurnRad = Math.PI / 16; // smooth turn rate
  const biasSign = legIndex % 2 === 0 ? 1 : -1;
  const maxBiasRad = clamp(totalDistanceM / 4_500_000, 0.03, 0.16); // 1.7deg..9.2deg
  const points: [number, number][] = [];
  points.push(start);

  let x = startX;
  let y = startY;
  let heading = Math.atan2(endY - startY, endX - startX);
  let prevRemainingM = totalDistanceM;

  for (let idx = 0; idx < targetSteps - 1; idx += 1) {
    const remainingX = endX - x;
    const remainingY = endY - y;
    const remainingM = Math.hypot(remainingX, remainingY);
    if (remainingM <= stepM * 1.05) break;

    const progress = clamp01(1 - remainingM / totalDistanceM);
    const desiredHeading = Math.atan2(remainingY, remainingX);
    const biasedHeading =
      desiredHeading + Math.sin(progress * Math.PI) * maxBiasRad * biasSign;
    const quantizedHeading =
      Math.round(biasedHeading / quantStepRad) * quantStepRad;
    const steeringDelta = clamp(
      wrapAngle(quantizedHeading - heading),
      -maxTurnRad,
      maxTurnRad,
    );
    heading += steeringDelta;

    x += Math.cos(heading) * stepM;
    y += Math.sin(heading) * stepM;

    const nextRemainingM = Math.hypot(endX - x, endY - y);
    if (nextRemainingM > prevRemainingM + stepM * 0.12) {
      // Snap back toward destination if quantized steering diverges.
      heading = desiredHeading;
      x += Math.cos(heading) * stepM * 0.75;
      y += Math.sin(heading) * stepM * 0.75;
    }
    prevRemainingM = Math.min(prevRemainingM, nextRemainingM);

    const [lon, lat] = unprojectXY(x, y, refLatRad);
    const prev = points[points.length - 1];
    if (!prev || Math.hypot(lon - prev[0], lat - prev[1]) > 1e-6) {
      points.push([lon, lat]);
    }
  }

  const last = points[points.length - 1];
  if (!last || Math.hypot(last[0] - end[0], last[1] - end[1]) > 1e-6) {
    points.push(end);
  }
  return points.length >= 2 ? points : [start, end];
}

export function buildPreviewRouteSegmentsFromNodes(nodes: PreviewRouteNode[]): PreviewDotSegment[] {
  if (nodes.length < 2) return [];

  const segments: PreviewDotSegment[] = [];
  for (let legIndex = 0; legIndex < nodes.length - 1; legIndex += 1) {
    const fromNode = nodes[legIndex];
    const toNode = nodes[legIndex + 1];
    const legPoints = buildRoadLikeLegPoints(
      [fromNode.lon, fromNode.lat],
      [toNode.lon, toNode.lat],
      legIndex,
    );
    if (legPoints.length < 2) continue;

    const startColor = fromNode.color;
    const endColor = toNode.color;
    const denom = Math.max(1, legPoints.length - 1);

    for (let segmentIndex = 0; segmentIndex < legPoints.length - 1; segmentIndex += 1) {
      if (segmentIndex % 2 !== 0) continue;
      const from = legPoints[segmentIndex];
      const to = legPoints[segmentIndex + 1];
      if (Math.hypot(to[0] - from[0], to[1] - from[1]) <= 1e-9) continue;

      const t = (segmentIndex + 0.5) / denom;
      segments.push({
        id: `preview-${legIndex + 1}-${segmentIndex + 1}`,
        legIndex,
        segmentIndex,
        fromNodeId: fromNode.id,
        toNodeId: toNode.id,
        from,
        to,
        color: interpolateHexColor(startColor, endColor, t),
      });
    }
  }

  return segments;
}

export function buildPreviewRouteSegments(
  origin: LatLng | null,
  destination: LatLng | null,
  dutyStops: DutyChainStop[],
): PreviewDotSegment[] {
  const nodes = buildPreviewRouteNodes(origin, destination, dutyStops);
  return buildPreviewRouteSegmentsFromNodes(nodes);
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
      label: 'End',
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
  const previewNodes = buildPreviewRouteNodes(
    origin,
    destination,
    stop ? [{ lat: stop.lat, lon: stop.lon, label: stop.label }] : [],
  );
  const colorById = new Map(previewNodes.map((node) => [node.id, node.color]));

  const out: PinDisplayNode[] = [];
  if (origin) {
    out.push({
      id: 'origin',
      kind: 'origin',
      lat: origin.lat,
      lon: origin.lon,
      label: labels.origin?.trim() || 'Start',
      order: 1,
      color: colorById.get('origin') ?? PREVIEW_START_COLOR,
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
      color: colorById.get('stop-1') ?? interpolateHexColor(PREVIEW_START_COLOR, PREVIEW_END_COLOR, 0.5),
    });
  }
  if (destination) {
    out.push({
      id: 'destination',
      kind: 'destination',
      lat: destination.lat,
      lon: destination.lon,
      label: labels.destination?.trim() || 'End',
      order: stop ? 3 : 2,
      color: colorById.get('destination') ?? PREVIEW_END_COLOR,
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
