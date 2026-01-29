import path from 'path';
import { fileURLToPath } from 'url';

/** @type {import('next').NextConfig} */
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',

  // Prevent Next/Turbopack from guessing a workspace root outside this folder
  // (you were seeing the warning about multiple lockfiles and it picking
  //  C:\\Users\\...\\package-lock.json as the root).
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
