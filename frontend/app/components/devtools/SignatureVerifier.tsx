'use client';

import { useState } from 'react';

import type {
  SignatureVerificationRequest,
  SignatureVerificationResponse,
} from '../../lib/types';

type Props = {
  loading: boolean;
  error: string | null;
  result: SignatureVerificationResponse | null;
  onVerify: (request: SignatureVerificationRequest) => Promise<void> | void;
};

const DEFAULT_PAYLOAD = JSON.stringify(
  {
    run_id: 'example-run-id',
    created_at: '2026-01-01T00:00:00Z',
  },
  null,
  2,
);

export default function SignatureVerifier({ loading, error, result, onVerify }: Props) {
  const [payloadText, setPayloadText] = useState(DEFAULT_PAYLOAD);
  const [signature, setSignature] = useState('');
  const [secret, setSecret] = useState('');
  const [localError, setLocalError] = useState<string | null>(null);

  async function verify() {
    try {
      setLocalError(null);
      let payload: SignatureVerificationRequest['payload'] = payloadText;
      const trimmed = payloadText.trim();
      if (trimmed.startsWith('{') || trimmed.startsWith('[') || trimmed.startsWith('"')) {
        payload = JSON.parse(trimmed) as SignatureVerificationRequest['payload'];
      }
      await onVerify({
        payload,
        signature,
        ...(secret.trim() ? { secret: secret.trim() } : {}),
      });
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : 'Failed to verify signature.');
    }
  }

  return (
    <div className="devCard">
      <div className="devCard__head">
        <h4 className="devCard__title">Signature Verifier</h4>
      </div>
      {error ? <div className="error">{error}</div> : null}
      {localError ? <div className="error">{localError}</div> : null}

      <div className="fieldLabel">Payload (JSON or String)</div>
      <textarea
        className="input devTextarea"
        value={payloadText}
        disabled={loading}
        onChange={(event) => setPayloadText(event.target.value)}
      />

      <div className="fieldLabel">Signature</div>
      <input
        className="input"
        value={signature}
        disabled={loading}
        onChange={(event) => setSignature(event.target.value)}
      />

      <div className="fieldLabel">Secret (optional)</div>
      <input
        className="input"
        value={secret}
        disabled={loading}
        onChange={(event) => setSecret(event.target.value)}
      />

      <div className="actionGrid actionGrid--single u-mt10">
        <button type="button" className="secondary" onClick={verify} disabled={loading || !signature.trim()}>
          {loading ? 'Verifying...' : 'Verify Signature'}
        </button>
      </div>

      {result ? <pre className="devPre">{JSON.stringify(result, null, 2)}</pre> : null}
    </div>
  );
}
