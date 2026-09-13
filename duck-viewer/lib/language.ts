export type Language = "en" | "zh";

export const LANGUAGE_STORAGE_KEY = "microduck.ui.language";

export function normalizeLanguage(value: unknown): Language {
  return value === "zh" ? "zh" : "en";
}

export function translate(language: Language, english: string, chinese: string): string {
  return language === "zh" ? chinese : english;
}

export function createLanguageStore(getStorage: () => Pick<Storage, "getItem" | "setItem">) {
  let current: Language | undefined;
  const listeners = new Set<() => void>();

  function getLanguage(): Language {
    if (current !== undefined) return current;
    try {
      return normalizeLanguage(getStorage().getItem(LANGUAGE_STORAGE_KEY));
    } catch {
      return "en";
    }
  }

  function synchronize(value: unknown) {
    current = normalizeLanguage(value);
    listeners.forEach((listener) => listener());
  }

  function setLanguage(language: Language) {
    try {
      getStorage().setItem(LANGUAGE_STORAGE_KEY, language);
    } catch {}
    synchronize(language);
  }

  function subscribe(listener: () => void) {
    listeners.add(listener);
    return () => { listeners.delete(listener); };
  }

  return { getLanguage, setLanguage, subscribe, synchronize };
}
