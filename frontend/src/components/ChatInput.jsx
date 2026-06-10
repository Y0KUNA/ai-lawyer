import React, { useCallback, useEffect, useRef, useState } from "react";
import { Box, IconButton, TextField } from "@mui/material";
import SendIcon from "@mui/icons-material/Send";

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState("");
  const inputRef = useRef();

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const send = useCallback(() => {
    const text = value.trim();
    if (!text) return;
    onSend(text);
    setValue("");
  }, [value, onSend]);

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <Box sx={{ display: "flex", gap: 1, alignItems: "center" }}>
      <TextField
        inputRef={inputRef}
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={onKeyDown}
        placeholder="Nhập câu hỏi pháp lý..."
        multiline
        minRows={1}
        maxRows={6}
        fullWidth
        variant="outlined"
        disabled={disabled}
        sx={{ borderRadius: 3 }}
      />

      <IconButton color="primary" onClick={send} disabled={disabled} aria-label="send">
        <SendIcon />
      </IconButton>
    </Box>
  );
}