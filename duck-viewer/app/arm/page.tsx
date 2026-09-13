"use client";

import dynamic from "next/dynamic";

const ArmStudio = dynamic(() => import("@/components/ArmStudio"), { ssr: false });

export default function ArmPage() {
  return <ArmStudio />;
}
