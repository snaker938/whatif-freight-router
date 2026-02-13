'use client';

import { useMemo, useRef } from 'react';

import type { TutorialLockScope, TutorialTargetRect } from '../lib/tutorial/types';

type TutorialMode = 'blocked' | 'chooser' | 'running' | 'completed';

type ChecklistItem = {
  actionId: string;
  label: string;
  details?: string;
  done: boolean;
  kind?: 'ui' | 'manual';
};

type OptionalDecisionState = {
  id: string;
  label: string;
  resolved: boolean;
  defaultLabel: string;
  actionTouched: boolean;
};

type Props = {
  open: boolean;
  mode: TutorialMode;
  isDesktop: boolean;
  hasSavedProgress: boolean;
  chapterTitle: string;
  chapterDescription: string;
  chapterIndex: number;
  chapterCount: number;
  stepTitle: string;
  stepWhat: string;
  stepImpact: string;
  stepIndex: number;
  stepCount: number;
  canGoNext: boolean;
  atStart: boolean;
  atEnd: boolean;
  checklist: ChecklistItem[];
  optionalDecision: OptionalDecisionState | null;
  targetRect: TutorialTargetRect | null;
  targetMissing: boolean;
  runningScope?: TutorialLockScope;
  onClose: () => void;
  onStartNew: () => void;
  onResume: () => void;
  onRestart: () => void;
  onBack: () => void;
  onNext: () => void;
  onFinish: () => void;
  onMarkManual: (actionId: string) => void;
  onUseOptionalDefault: (optionalDecisionId: string) => void;
};

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export default function TutorialOverlay({
  open,
  mode,
  isDesktop,
  hasSavedProgress,
  chapterTitle,
  chapterDescription,
  chapterIndex,
  chapterCount,
  stepTitle,
  stepWhat,
  stepImpact,
  stepIndex,
  stepCount,
  canGoNext,
  atStart,
  atEnd,
  checklist,
  optionalDecision,
  targetRect,
  targetMissing,
  runningScope = 'free',
  onClose,
  onStartNew,
  onResume,
  onRestart,
  onBack,
  onNext,
  onFinish,
  onMarkManual,
  onUseOptionalDefault,
}: Props) {
  const suppressBackdropCloseRef = useRef(false);

  const layout = useMemo(() => {
    const vw = typeof window !== 'undefined' ? window.innerWidth : 1440;
    const vh = typeof window !== 'undefined' ? window.innerHeight : 900;
    const cardWidth = Math.min(500, vw - 28);

    if (!targetRect || mode !== 'running') {
      return {
        cardStyle: {
          left: clamp((vw - cardWidth) / 2, 12, vw - cardWidth - 12),
          top: clamp(vh * 0.15, 12, vh - 380),
          width: cardWidth,
        },
        arrowStyle: null as { left: number; top: number } | null,
        arrowClass: 'isHidden',
      };
    }

    const targetRight = targetRect.left + targetRect.width;
    const targetMidY = targetRect.top + targetRect.height / 2;

    const preferRight = targetRight + 24 + cardWidth < vw - 16;
    const left = preferRight
      ? targetRight + 24
      : clamp(targetRect.left - cardWidth - 24, 12, vw - cardWidth - 12);
    const top = clamp(targetMidY - 220, 12, vh - 420);

    return {
      cardStyle: { left, top, width: cardWidth },
      arrowStyle: { left, top },
      arrowClass: preferRight ? 'isLeftAnchor' : 'isRightAnchor',
    };
  }, [mode, targetRect]);

  if (!open) return null;
  const mapFocusedRunning = mode === 'running' && runningScope === 'map_only';

  function handleBackdropClick() {
    if (mode === 'running') {
      return;
    }
    if (suppressBackdropCloseRef.current) {
      suppressBackdropCloseRef.current = false;
      return;
    }
    onClose();
  }

  return (
    <div
      className={`tutorialOverlay ${mode === 'running' ? 'isRunning' : ''} ${mapFocusedRunning ? 'isMapFocused' : ''}`.trim()}
      role="dialog"
      aria-modal="true"
      aria-label="Guided frontend tutorial"
    >
      <div className="tutorialOverlay__backdrop" onClick={handleBackdropClick} />

      {mode === 'running' && targetRect && !mapFocusedRunning ? (
        <>
          <div
            className="tutorialOverlay__spotlight"
            style={{
              top: targetRect.top - 6,
              left: targetRect.left - 6,
              width: targetRect.width + 12,
              height: targetRect.height + 12,
            }}
          />
          <div className={`tutorialOverlay__arrow ${layout.arrowClass}`} style={layout.arrowStyle ?? undefined} />
        </>
      ) : null}

      <div
        className="tutorialOverlay__card tutorialOverlay__card--guided"
        style={layout.cardStyle}
        onMouseDown={() => {
          suppressBackdropCloseRef.current = true;
        }}
        onMouseUp={() => {
          window.setTimeout(() => {
            suppressBackdropCloseRef.current = false;
          }, 0);
        }}
        onClick={(event) => {
          event.stopPropagation();
        }}
      >
        {mode === 'blocked' ? (
          <>
            <div className="tutorialOverlay__badge">Desktop only</div>
            <h2 className="tutorialOverlay__title">Tutorial requires desktop width</h2>
            <p className="tutorialOverlay__body">
              This full guided tutorial is currently designed for desktop-only interaction. Expand your
              viewport to continue.
            </p>
            <div className="tutorialOverlay__actions">
              <button type="button" className="primary" onClick={onClose}>
                Close
              </button>
            </div>
          </>
        ) : null}

        {mode === 'chooser' ? (
          <>
            <div className="tutorialOverlay__badge">Guided tutorial</div>
            <h2 className="tutorialOverlay__title">Resume or restart tutorial</h2>
            <p className="tutorialOverlay__body">
              {hasSavedProgress
                ? 'You have unfinished tutorial progress. Resume where you left off or restart from chapter one.'
                : 'Start the full guided walkthrough from chapter one.'}
            </p>
            <div className="tutorialOverlay__actions">
              {hasSavedProgress ? (
                <button type="button" className="secondary" onClick={onResume}>
                  Resume
                </button>
              ) : null}
              <button
                type="button"
                className="primary"
                onClick={hasSavedProgress ? onRestart : onStartNew}
              >
                {hasSavedProgress ? 'Restart' : 'Start tutorial'}
              </button>
            </div>
            <div className="tutorialOverlay__footer">
              <button type="button" className="ghostButton" onClick={onClose}>
                Close
              </button>
            </div>
          </>
        ) : null}

        {mode === 'completed' ? (
          <>
            <div className="tutorialOverlay__badge">Completed</div>
            <h2 className="tutorialOverlay__title">Tutorial completed</h2>
            <p className="tutorialOverlay__body">
              You can restart the full walkthrough at any time from the Setup card.
            </p>
            <div className="tutorialOverlay__actions">
              <button type="button" className="secondary" onClick={onRestart}>
                Restart tutorial
              </button>
              <button type="button" className="primary" onClick={onClose}>
                Close
              </button>
            </div>
          </>
        ) : null}

        {mode === 'running' ? (
          <>
            <div className="tutorialOverlay__badge">
              Chapter {chapterIndex} / {chapterCount}
            </div>
            <div className="tutorialOverlay__chapterTitle">{chapterTitle}</div>
            <div className="tutorialOverlay__chapterHint">{chapterDescription}</div>
            <div className="tutorialOverlay__stepCounter">
              Step {stepIndex} / {stepCount}
            </div>

            <h2 className="tutorialOverlay__title">{stepTitle}</h2>
            <p className="tutorialOverlay__body">
              <strong>What to do:</strong> {stepWhat}
            </p>
            <p className="tutorialOverlay__body">
              <strong>How results change:</strong> {stepImpact}
            </p>

            {targetMissing ? (
              <div className="warningPanel" role="status" aria-live="polite">
                Target control is not currently visible for this state. Complete available requirements
                below and continue.
              </div>
            ) : null}

            {checklist.length > 0 ? (
              <ul className="tutorialChecklist">
                {checklist.map((item) => (
                  <li key={item.actionId} className={`tutorialChecklist__item ${item.done ? 'isDone' : ''}`}>
                    <span className="tutorialChecklist__icon" aria-hidden="true">
                      {item.done ? '✓' : '•'}
                    </span>
                    <span className="tutorialChecklist__labelWrap">
                      <span className="tutorialChecklist__label">{item.label}</span>
                      {item.details ? (
                        <span className="tutorialChecklist__details">{item.details}</span>
                      ) : null}
                    </span>
                    <span className={`tutorialChecklist__state ${item.done ? 'isDone' : 'isPending'}`}>
                      {item.done ? 'Done' : 'Pending'}
                    </span>
                    {!item.done && item.kind === 'manual' ? (
                  <button
                    type="button"
                    className="ghostButton tutorialChecklist__manualBtn"
                    onClick={() => onMarkManual(item.actionId)}
                      >
                        Mark done
                      </button>
                    ) : null}
                  </li>
                ))}
              </ul>
            ) : null}

            {optionalDecision ? (
              <div className="tutorialOptional">
                <div className="tutorialOptional__title">Optional field decision</div>
                <div className="tutorialOptional__body">{optionalDecision.label}</div>
                {optionalDecision.resolved || optionalDecision.actionTouched ? (
                  <div className="tutorialOptional__resolved">Resolved</div>
                ) : (
                  <button
                    type="button"
                    className="secondary"
                    onClick={() => onUseOptionalDefault(optionalDecision.id)}
                  >
                    {optionalDecision.defaultLabel}
                  </button>
                )}
              </div>
            ) : null}

            <div className="tutorialOverlay__actions">
              <button type="button" className="secondary" onClick={onBack} disabled={atStart}>
                Back
              </button>
              {atEnd ? (
                <button type="button" className="primary" onClick={onFinish} disabled={!canGoNext}>
                  Finish tutorial
                </button>
              ) : (
                <button type="button" className="primary" onClick={onNext} disabled={!canGoNext}>
                  Next
                </button>
              )}
            </div>

            <div className="tutorialOverlay__footer">
              {!isDesktop ? null : (
                <button type="button" className="ghostButton" onClick={onClose}>
                  Close and keep progress
                </button>
              )}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
