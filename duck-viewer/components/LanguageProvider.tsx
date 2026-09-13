"use client";

import { createContext, useContext, useEffect, useMemo, useSyncExternalStore, type ReactNode } from "react";
import { createLanguageStore, LANGUAGE_STORAGE_KEY, translate, type Language } from "@/lib/language";
import styles from "./LanguageProvider.module.css";

const store = createLanguageStore(() => window.localStorage);
const serverLanguage = (): Language => "en";
const LanguageContext = createContext<{
  language: Language;
  setLanguage: (language: Language) => void;
  t: (english: string, chinese: string) => string;
} | null>(null);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const language = useSyncExternalStore(store.subscribe, store.getLanguage, serverLanguage);

  useEffect(() => {
    document.documentElement.lang = language === "zh" ? "zh-CN" : "en";
  }, [language]);

  useEffect(() => {
    function onStorage(event: StorageEvent) {
      if (event.key === LANGUAGE_STORAGE_KEY || event.key === null) {
        try {
          if (event.storageArea !== window.localStorage) return;
        } catch {
          return;
        }
        store.synchronize(event.newValue);
      }
    }
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const value = useMemo(() => ({
    language,
    setLanguage: store.setLanguage,
    t: (english: string, chinese: string) => translate(language, english, chinese),
  }), [language]);

  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>;
}

export function useLanguage() {
  const value = useContext(LanguageContext);
  if (!value) throw new Error("useLanguage requires LanguageProvider");
  return value;
}

export function LanguageToggle() {
  const { language, setLanguage, t } = useLanguage();
  return (
    <div className={styles.toggle} role="group" aria-label={t("Interface language", "界面语言")} data-testid="language-toggle">
      <button type="button" lang="en" aria-pressed={language === "en"} onClick={() => setLanguage("en")}>English</button>
      <button type="button" lang="zh-CN" aria-pressed={language === "zh"} onClick={() => setLanguage("zh")}>中文</button>
    </div>
  );
}
