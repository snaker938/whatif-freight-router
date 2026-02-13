'use client';

import { useId, useState } from 'react';

type Props = {
  text: string;
  id?: string;
};

export default function FieldInfo({ text, id }: Props) {
  const autoId = useId();
  const tooltipId = id ?? `field-info-${autoId.replace(/[:]/g, '-')}`;
  const [open, setOpen] = useState(false);

  return (
    <span className="fieldInfoWrap" data-open={open ? 'true' : 'false'}>
      <button
        type="button"
        className="fieldInfo"
        aria-label={`Field Information: ${text}`}
        aria-describedby={tooltipId}
        aria-expanded={open}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onKeyDown={(event) => {
          if (event.key === 'Escape') {
            setOpen(false);
          }
        }}
      >
        <span aria-hidden="true">i</span>
      </button>
      <span id={tooltipId} className="fieldInfo__tooltip" role="note" aria-hidden={!open}>
        {text}
      </span>
    </span>
  );
}
