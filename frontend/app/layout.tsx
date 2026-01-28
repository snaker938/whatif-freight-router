import type { Metadata } from 'next';
import { Inter, Space_Grotesk } from 'next/font/google';

import 'leaflet/dist/leaflet.css';
import './global.css';

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

const space = Space_Grotesk({
  subsets: ['latin'],
  variable: '--font-space',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'Carbon-Aware Freight Router',
  description: 'Multi-objective freight routing demo',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${space.variable}`}>
      <body>{children}</body>
    </html>
  );
}
