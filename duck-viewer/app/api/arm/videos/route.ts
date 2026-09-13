import { listArmVideoLibrary } from "@/lib/arm-video-library";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return Response.json(await listArmVideoLibrary(), {
    headers: { "Cache-Control": "no-store" },
  });
}
