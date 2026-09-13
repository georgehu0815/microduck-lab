import { serveArmMedia } from "@/lib/arm-video-library";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  return serveArmMedia(request);
}

export async function HEAD(request: Request) {
  return serveArmMedia(request);
}
