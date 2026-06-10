import React, { useEffect, useMemo, useRef } from "react";
import { Box, Typography } from "@mui/material";
import MessageBubble from "./MessageBubble";

export default function ChatWindow({ conversation }) {
  const bottomRef = useRef();

  const lastMessageSignature = useMemo(() => {
    if (!conversation?.messages?.length) return "";
    const last = conversation.messages[conversation.messages.length - 1];
    return `${conversation.messages.length}:${last?.id || ""}:${last?.content?.length || 0}`;
  }, [conversation]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [lastMessageSignature]);

  if (!conversation) return null;

  return (
    <Box sx={{ flex: 1, display: "flex", flexDirection: "column", p: 3, overflow: "hidden" }}>
      {conversation.messages.length === 0 ? (
        <Box sx={{ textAlign: "center", mt: 8 }}>
          <Typography variant="h5" gutterBottom>
            Chào mừng tới AI Lawyer
          </Typography>
          <Typography color="text.secondary">
            Bắt đầu cuộc trò chuyện để nhận phân tích pháp lý chi tiết. Khu vực bên phải sẽ hiển thị nguồn tham khảo
            pháp luật trong tương lai.
          </Typography>
        </Box>
      ) : (
        <Box sx={{ overflow: "auto", flex: 1, pr: 1 }}>
          {conversation.messages.map((m, idx) => (
            <MessageBubble key={m.id || idx} message={m} />
          ))}
          <div ref={bottomRef} />
        </Box>
      )}
    </Box>
  );
}
