"use client";

import { useEffect, useRef, useState } from "react";

import { useLanguage } from "./LanguageProvider";
import styles from "./ContactDialog.module.css";

const CONTACT_EMAIL = "bochuxt7@gmail.com";
const GMAIL_COMPOSE_URL = `https://mail.google.com/mail/?view=cm&fs=1&to=${encodeURIComponent(CONTACT_EMAIL)}`;

type CopyState = "idle" | "copied" | "denied";

export default function ContactDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const { t } = useLanguage();
  const dialogRef = useRef<HTMLDialogElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);
  const emailRef = useRef<HTMLInputElement>(null);
  const restoreFocusRef = useRef<HTMLElement | null>(null);
  const [copyState, setCopyState] = useState<CopyState>("idle");

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;

    if (open) {
      restoreFocusRef.current = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null;
      if (!dialog.open) dialog.showModal();
      closeButtonRef.current?.focus();
      return;
    }

    if (dialog.open) dialog.close();
    restoreFocusRef.current?.focus();
    restoreFocusRef.current = null;
  }, [open]);

  async function copyEmail() {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
      await navigator.clipboard.writeText(CONTACT_EMAIL);
      setCopyState("copied");
    } catch {
      setCopyState("denied");
      emailRef.current?.focus();
      emailRef.current?.select();
    }
  }

  function closeDialog() {
    setCopyState("idle");
    onClose();
  }

  return (
    <dialog
      ref={dialogRef}
      className={styles.dialog}
      aria-labelledby="contact-dialog-title"
      aria-describedby="contact-dialog-description"
      onCancel={(event) => {
        event.preventDefault();
        closeDialog();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) closeDialog();
      }}
    >
      <div className={styles.panel}>
        <div className={styles.heading}>
          <div>
            <p className={styles.eyebrow}>MICRODUCK STUDIO</p>
            <h2 id="contact-dialog-title">{t("Contact", "联系")}</h2>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className={styles.closeButton}
            onClick={closeDialog}
            aria-label={t("Close contact dialog", "关闭联系对话框")}
          >
            ×
          </button>
        </div>

        <p id="contact-dialog-description" className={styles.description}>
          {t(
            "Choose how you want to write. Nothing is sent or opened until you select an option.",
            "请选择写信方式。只有在您选择相应选项后，才会打开邮件服务；不会自动发送任何内容。"
          )}
        </p>

        <label className={styles.emailField}>
          <span>{t("Email address", "邮箱地址")}</span>
          <input
            ref={emailRef}
            type="text"
            value={CONTACT_EMAIL}
            readOnly
            onFocus={(event) => event.currentTarget.select()}
            aria-describedby={copyState === "denied" ? "contact-copy-status" : undefined}
          />
        </label>

        <div className={styles.actions}>
          <button type="button" className={styles.copyButton} onClick={() => void copyEmail()}>
            {copyState === "copied" ? t("Copied", "已复制") : t("Copy address", "复制地址")}
          </button>
          <a className={styles.writeButton} href={`mailto:${CONTACT_EMAIL}`}>
            {t("Write email", "写邮件")}
          </a>
          <a
            className={styles.gmailButton}
            href={GMAIL_COMPOSE_URL}
            target="_blank"
            rel="noopener noreferrer"
          >
            {t("Open Gmail", "打开 Gmail")}
          </a>
        </div>

        <p
          id="contact-copy-status"
          className={copyState === "denied" ? styles.copyError : styles.copyStatus}
          role="status"
          aria-live="polite"
        >
          {copyState === "copied"
            ? t("Email address copied.", "邮箱地址已复制。")
            : copyState === "denied"
              ? t(
                  "Clipboard access was denied. The address is selected above so you can copy it manually.",
                  "剪贴板访问被拒绝。上方地址已选中，您可以手动复制。"
                )
              : ""}
        </p>
      </div>
    </dialog>
  );
}
