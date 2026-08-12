const API_PROXY_TARGET = process.env.API_PROXY_TARGET ?? "http://127.0.0.1:8000";

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Milestone 14: standalone output for a small, self-contained production
  // Docker image (.next/standalone + .next/static + public; no devDependencies
  // or full node_modules in the runtime stage).
  output: "standalone",
  async rewrites() {
    return [{ source: "/v1/:path*", destination: `${API_PROXY_TARGET}/v1/:path*` }];
  },
};

export default nextConfig;
