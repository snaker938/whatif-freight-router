'use client';

import { useEffect, useMemo, useState } from 'react';

type Props = {
  open: boolean;
  onClose: (markSeen: boolean) => void;
};

const STEPS = [
  {
    title: 'Set map pins',
    body: 'Click the map to place Start and Destination pins. You can drag, swap, or clear pins at any time.',
  },
  {
    title: 'Compute routes',
    body: 'Set preferences and press Compute Pareto to generate candidate routes and choose a selected route.',
  },
  {
    title: 'Compare scenarios',
    body: 'Use Compare scenarios to see No/Partial/Full sharing deltas side by side.',
  },
  {
    title: 'Optimize departures',
    body: 'Run departure optimization with optional arrival constraints to find the best departure slot.',
  },
  {
    title: 'Review artifacts',
    body: 'Use run artifacts endpoints to download JSON/CSV/GeoJSON/PDF outputs for reproducibility and reporting.',
  },
];

export default function TutorialOverlay({ open, onClose }: Props) {
  const [stepIdx, setStepIdx] = useState(0);

  useEffect(() => {
    if (open) {
      setStepIdx(0);
    }
  }, [open]);

  const step = useMemo(() => STEPS[Math.max(0, Math.min(stepIdx, STEPS.length - 1))], [stepIdx]);
  const atStart = stepIdx <= 0;
  const atEnd = stepIdx >= STEPS.length - 1;

  if (!open) return null;

  return (
    <div className="tutorialOverlay" role="dialog" aria-modal="true" aria-label="Interactive tutorial">
      <div className="tutorialOverlay__backdrop" onClick={() => onClose(true)} />
      <div className="tutorialOverlay__card">
        <div className="tutorialOverlay__badge">
          Step {stepIdx + 1} of {STEPS.length}
        </div>
        <h2 className="tutorialOverlay__title">{step.title}</h2>
        <p className="tutorialOverlay__body">{step.body}</p>

        <div className="tutorialOverlay__actions">
          <button
            className="secondary"
            onClick={() => setStepIdx((prev) => Math.max(0, prev - 1))}
            disabled={atStart}
          >
            Back
          </button>
          {!atEnd ? (
            <button className="primary" onClick={() => setStepIdx((prev) => Math.min(prev + 1, STEPS.length - 1))}>
              Next
            </button>
          ) : (
            <button
              className="primary"
              onClick={() => {
                setStepIdx(0);
                onClose(true);
              }}
            >
              Finish tutorial
            </button>
          )}
        </div>

        <div className="tutorialOverlay__footer">
          <button className="ghostButton" onClick={() => onClose(true)}>
            Skip and close
          </button>
        </div>
      </div>
    </div>
  );
}
