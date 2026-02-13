'use client';

import { useMemo, useState } from 'react';

import type { PinDisplayNode, PinSelectionId } from '../lib/types';

type Props = {
  nodes: PinDisplayNode[];
  selectedPinId: PinSelectionId | null;
  disabled: boolean;
  hasStop: boolean;
  canAddStop: boolean;
  oneStopHint?: string | null;
  onSelectPin: (id: PinSelectionId) => void;
  onRenameStart: (name: string) => void;
  onRenameDestination: (name: string) => void;
  onRenameStop: (name: string) => void;
  onAddStop: () => void;
  onDeleteStop: () => void;
  onSwapPins: () => void;
  onClearPins: () => void;
};

function fmtCoord(value: number): string {
  return Number.isFinite(value) ? value.toFixed(5) : String(value);
}

async function copyToClipboard(text: string): Promise<boolean> {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
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

export default function PinManager({
  nodes,
  selectedPinId,
  disabled,
  hasStop,
  canAddStop,
  oneStopHint = null,
  onSelectPin,
  onRenameStart,
  onRenameDestination,
  onRenameStop,
  onAddStop,
  onDeleteStop,
  onSwapPins,
  onClearPins,
}: Props) {
  const [copiedNodeId, setCopiedNodeId] = useState<string | null>(null);

  const originNode = useMemo(() => nodes.find((node) => node.id === 'origin') ?? null, [nodes]);
  const stopNode = useMemo(() => nodes.find((node) => node.id === 'stop-1') ?? null, [nodes]);
  const destinationNode = useMemo(
    () => nodes.find((node) => node.id === 'destination') ?? null,
    [nodes],
  );

  function renderNodeRow(node: PinDisplayNode, label: string, onRename: (name: string) => void) {
    const isSelected = selectedPinId === node.id;
    return (
      <li key={node.id} className={`pinManager__row ${isSelected ? 'isSelected' : ''}`}>
        <button
          type="button"
          className="pinManager__nodeBtn"
          onClick={() => onSelectPin(node.id)}
          disabled={disabled}
          aria-pressed={isSelected}
          aria-label={`${isSelected ? 'Deselect' : 'Select'} ${label} pin`}
          data-tutorial-action="pins.sidebar_select"
        >
          <span className={`pinManager__dot pinManager__dot--${node.kind}`} />
          <span className="pinManager__nodeText">
            <span className="pinManager__nodeTitle">{label}</span>
            <span className="pinManager__nodeCoords">
              {fmtCoord(node.lat)}, {fmtCoord(node.lon)}
            </span>
          </span>
        </button>

        <div className="pinManager__inlineActions">
          <input
            className="input pinManager__nameInput"
            value={node.label}
            onChange={(event) => onRename(event.target.value)}
            placeholder={label}
            disabled={disabled}
            aria-label={`${label} name`}
          />
          <button
            type="button"
            className="secondary pinManager__copyBtn"
            disabled={disabled}
            onClick={async () => {
              const copied = await copyToClipboard(`${fmtCoord(node.lat)}, ${fmtCoord(node.lon)}`);
              if (!copied) return;
              setCopiedNodeId(node.id);
              window.setTimeout(() => {
                setCopiedNodeId((prev) => (prev === node.id ? null : prev));
              }, 900);
            }}
          >
            {copiedNodeId === node.id ? 'Copied' : 'Copy'}
          </button>
        </div>
      </li>
    );
  }

  return (
    <section className="card pinManager" data-tutorial-id="pins.section">
      <div className="sectionTitle">Pins & stops</div>
      <div className="sectionHint">Manage start, destination, and one optional stop from one place.</div>

      <div className="pinManager__rail" aria-hidden="true">
        <span>Start</span>
        <span className="pinManager__arrow">→</span>
        <span>Stop #1</span>
        <span className="pinManager__arrow">→</span>
        <span>Destination</span>
      </div>

      <ul className="pinManager__list">
        {originNode ? renderNodeRow(originNode, 'Start', onRenameStart) : null}
        {stopNode ? renderNodeRow(stopNode, 'Stop #1', onRenameStop) : null}
        {destinationNode ? renderNodeRow(destinationNode, 'Destination', onRenameDestination) : null}
      </ul>

      <div className="pinManager__actions">
        <button
          type="button"
          className="secondary"
          disabled={disabled || !canAddStop}
          onClick={onAddStop}
          title={!canAddStop ? 'Set both Start and Destination to add a stop.' : 'Add midpoint stop'}
          data-tutorial-action="pins.add_stop"
        >
          {hasStop ? 'Replace stop' : 'Add stop'}
        </button>
        <button
          type="button"
          className="secondary"
          disabled={disabled || !hasStop}
          onClick={onDeleteStop}
          title={!hasStop ? 'No stop exists to delete.' : 'Delete stop'}
          data-tutorial-action="pins.delete_stop"
        >
          Delete stop
        </button>
        <button type="button" className="secondary" disabled={disabled} onClick={onSwapPins}>
          Swap start/end
        </button>
        <button type="button" className="secondary" disabled={disabled} onClick={onClearPins}>
          Clear pins
        </button>
      </div>

      <div className="pinManager__reorder">
        <button
          type="button"
          className="secondary"
          disabled={true}
          title="Unavailable in one-stop mode"
          aria-disabled="true"
        >
          Move stop up
        </button>
        <button
          type="button"
          className="secondary"
          disabled={true}
          title="Unavailable in one-stop mode"
          aria-disabled="true"
        >
          Move stop down
        </button>
      </div>
      <div className="tiny">Reorder is disabled in one-stop mode. Enable when multi-stop support is added.</div>
      {oneStopHint ? <div className="pinManager__hintError">{oneStopHint}</div> : null}
    </section>
  );
}
