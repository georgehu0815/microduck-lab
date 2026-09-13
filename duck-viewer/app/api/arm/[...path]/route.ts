import {
  ARM_GET_PATHS,
  ARM_POST_PATHS,
  handleArmGet,
  handleArmPost,
} from "@/lib/arm-proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type ArmRouteContext = {
  params: Promise<{ path: string[] }>;
};

function pathFrom(context: ArmRouteContext) {
  return context.params.then(({ path }) => path.join("/"));
}

export async function GET(
  request: Request,
  context: ArmRouteContext
) {
  const path = await pathFrom(context);
  if (!ARM_GET_PATHS.has(path)) {
    return Response.json({ error: "Unknown Arm API path." }, { status: 404 });
  }
  return handleArmGet(request, path);
}

export async function POST(
  request: Request,
  context: ArmRouteContext
) {
  const path = await pathFrom(context);
  if (!ARM_POST_PATHS.has(path)) {
    return Response.json({ error: "Unknown Arm API path." }, { status: 404 });
  }
  return handleArmPost(request, path);
}
