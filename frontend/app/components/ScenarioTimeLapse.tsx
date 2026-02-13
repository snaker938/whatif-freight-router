'use client';

import { useEffect, useMemo, useRef, useState } from 'react';

import CollapsibleCard from './CollapsibleCard';
import FieldInfo from './FieldInfo';
import {
  SIDEBAR_DROPDOWN_OPTIONS_HELP,
  SIDEBAR_FIELD_HELP,
  SIDEBAR_SECTION_HINTS,
} from '../lib/sidebarHelpText';
import type { LatLng, RouteOption } from '../lib/types';

type Props = {
  route: RouteOption | null;
  onPositionChange: (position: LatLng | null) => void;
};

const PLAYBACK_SPEEDS = [0.5, 1, 2, 4];

function pointAlongRoute(coords: [number, number][], progress: number): LatLng | null {
  if (coords.length < 2) {
    return null;
  }

  const clamped = Math.max(0, Math.min(1, progress));
  if (clamped <= 0) {
    const [lon, lat] = coords[0];
    return { lat, lon };
  }
  if (clamped >= 1) {
    const [lon, lat] = coords[coords.length - 1];
    return { lat, lon };
  }

  const lengths: number[] = [0];
  let total = 0;
  for (let i = 1; i < coords.length; i += 1) {
    const [prevLon, prevLat] = coords[i - 1];
    const [lon, lat] = coords[i];
    const dx = lon - prevLon;
    const dy = lat - prevLat;
    total += Math.hypot(dx, dy);
    lengths.push(total);
  }

  if (total <= 0) {
    const [lon, lat] = coords[0];
    return { lat, lon };
  }

  const target = clamped * total;
  let idx = 1;
  while (idx < lengths.length && lengths[idx] < target) {
    idx += 1;
  }

  const i = Math.min(idx, coords.length - 1);
  const segStartDist = lengths[i - 1];
  const segEndDist = lengths[i];
  const segSpan = Math.max(segEndDist - segStartDist, 1e-9);
  const ratio = (target - segStartDist) / segSpan;

  const [startLon, startLat] = coords[i - 1];
  const [endLon, endLat] = coords[i];
  const lon = startLon + (endLon - startLon) * ratio;
  const lat = startLat + (endLat - startLat) * ratio;
  return { lat, lon };
}

export default function ScenarioTimeLapse({ route, onPositionChange }: Props) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [speed, setSpeed] = useState(1);
  const lastFrameRef = useRef<number | null>(null);

  const coords = useMemo(() => route?.geometry?.coordinates ?? [], [route]);
  const durationS = route?.metrics?.duration_s ?? 0;

  useEffect(() => {
    setIsPlaying(false);
    setProgress(0);
    lastFrameRef.current = null;
  }, [route?.id]);

  useEffect(() => {
    if (!route || coords.length < 2) {
      onPositionChange(null);
      return;
    }
    onPositionChange(pointAlongRoute(coords, progress));
  }, [coords, onPositionChange, progress, route]);

  useEffect(() => {
    if (!isPlaying) {
      lastFrameRef.current = null;
      return;
    }

    let rafId = 0;
    const compressedDurationS = Math.max(6, Math.min(90, durationS / 20));

    const tick = (ts: number) => {
      const prevTs = lastFrameRef.current ?? ts;
      const dt = Math.max(0, (ts - prevTs) / 1000);
      lastFrameRef.current = ts;

      setProgress((prev) => {
        const next = Math.min(1, prev + (dt * speed) / compressedDurationS);
        return next;
      });

      rafId = window.requestAnimationFrame(tick);
    };

    rafId = window.requestAnimationFrame(tick);
    return () => {
      window.cancelAnimationFrame(rafId);
      lastFrameRef.current = null;
    };
  }, [durationS, isPlaying, speed]);

  useEffect(() => {
    if (progress >= 1 && isPlaying) {
      setIsPlaying(false);
    }
  }, [isPlaying, progress]);

  if (!route || coords.length < 2) {
    return (
      <CollapsibleCard
        title="Scenario time-lapse"
        hint={SIDEBAR_SECTION_HINTS.scenarioTimeLapse}
        dataTutorialId="timelapse.section"
      >
        <div className="helper">Compute and select a route to play a route animation.</div>
      </CollapsibleCard>
    );
  }

  const elapsedS = durationS * progress;

  return (
    <CollapsibleCard
      title="Scenario time-lapse"
      hint={SIDEBAR_SECTION_HINTS.scenarioTimeLapse}
      dataTutorialId="timelapse.section"
    >
      <div className="sectionTitleRow">
        <div className="routeCard__pill">{(progress * 100).toFixed(0)}%</div>
      </div>

      <div className="row">
        <button
          className="secondary"
          onClick={() => setIsPlaying((prev) => !prev)}
          data-tutorial-action={isPlaying ? 'timelapse.pause' : 'timelapse.play'}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          className="secondary"
          onClick={() => {
            setIsPlaying(false);
            setProgress(0);
          }}
          data-tutorial-action="timelapse.reset"
        >
          Reset
        </button>
      </div>

      <div className="fieldLabelRow">
        <label className="fieldLabel" htmlFor="time-lapse-speed">
          Playback speed
        </label>
        <FieldInfo text={SIDEBAR_FIELD_HELP.playbackSpeed} />
      </div>
      <select
        id="time-lapse-speed"
        className="input"
        value={String(speed)}
        onChange={(event) => setSpeed(Number(event.target.value))}
        data-tutorial-action="timelapse.speed_select"
      >
        {PLAYBACK_SPEEDS.map((value) => (
          <option key={value} value={String(value)}>
            x{value}
          </option>
        ))}
      </select>
      <div className="dropdownOptionsHint">{SIDEBAR_DROPDOWN_OPTIONS_HELP.timeLapseSpeed}</div>

      <label className="fieldLabel" htmlFor="time-lapse-progress">
        Timeline scrubber
      </label>
      <input
        id="time-lapse-progress"
        type="range"
        min={0}
        max={1000}
        value={Math.round(progress * 1000)}
        onChange={(event) => setProgress(Number(event.target.value) / 1000)}
        data-tutorial-action="timelapse.scrubber_input"
      />

      <div className="tiny">
        Elapsed {(elapsedS / 60).toFixed(1)} min of {(durationS / 60).toFixed(1)} min
      </div>
    </CollapsibleCard>
  );
}
