"use client";

import { useEffect } from "react";

export default function ViewerPage() {
  useEffect(() => {
    const url = new URL(window.location.href);
    const basePath = url.pathname.replace(/\/viewer\/?$/, "");
    const destination = `${basePath || ""}/${url.search}`;
    window.location.replace(destination);
  }, []);

  return null;
}
