# syntax=docker/dockerfile:1

FROM node:24.13.0-alpine AS base
WORKDIR /app

# pnpm via Corepack (packageManager field in package.json pins the version)
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable

FROM base AS builder

# Install deps first for caching
COPY package.json pnpm-lock.yaml* ./

# Install dependencies (using a cache for the pnpm store when BuildKit is enabled)
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile --store-dir /pnpm/store

# Now copy the rest of the source and build
COPY . .
RUN pnpm build

FROM node:24.13.0-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
ENV NEXT_TELEMETRY_DISABLED=1

# Next.js standalone output
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000
CMD ["node", "server.js"]
