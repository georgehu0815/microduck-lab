import type { Metadata } from "next";
import ClassroomPage from "@/components/ClassroomPage";

export const metadata: Metadata = {
  title: "Classroom | Microduck Studio",
  description: "Eight bilingual Microduck lesson decks, static slide previews, downloads, and recorded examples.",
};

export default function Page() {
  return <ClassroomPage />;
}
