import { spawnSync } from 'node:child_process';

const variable = 'NEXT_TURBOPACK_EXPERIMENTAL_USE_SYSTEM_TLS_CERTS';

if (process.env[variable] !== '1') {
  throw new Error(`${variable} was not set at the Next launch boundary`);
}

const child = spawnSync(
  process.execPath,
  ['-e', `process.stdout.write(process.env[${JSON.stringify(variable)}] ?? '')`],
  {
    env: process.env,
    encoding: 'utf8',
  },
);

if (child.error) {
  throw child.error;
}

if (child.status !== 0 || child.stdout !== '1') {
  throw new Error(
    `${variable} did not reach the spawned command (status=${child.status}, value=${JSON.stringify(child.stdout)})`,
  );
}

console.log(`${variable}=1 reached the spawned command`);
