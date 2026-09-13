import type { NextConfig } from "next";

const isStaticExport = process.env.DUCK_VIEWER_STATIC_EXPORT === "1";
const pagesBasePath = process.env.DUCK_VIEWER_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  allowedDevOrigins: ["127.0.0.1", "localhost"],
  ...(isStaticExport
    ? {
        output: "export" as const,
        basePath: pagesBasePath,
        assetPrefix: pagesBasePath,
        trailingSlash: true,
        images: { unoptimized: true },
      }
    : {}),
};

export default nextConfig;
