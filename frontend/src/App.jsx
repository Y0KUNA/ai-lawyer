import React, { useEffect, useMemo, useState } from "react";
import { ThemeProvider } from "@mui/material/styles";
import { Box, CssBaseline } from "@mui/material";
import theme from "./theme";
import Sidebar from "./components/SideBar";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import LoadingIndicator from "./components/LoadingIndicator";

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [loading, setLoading] = useState(false);

  const activeConversation = useMemo(() => {
    return conversations.find((c) => c.id === activeId) || null;
  }, [conversations, activeId]);

  useEffect(() => {
    if (conversations.length === 0) {
      const id = Date.now().toString();
      setConversations([
        {
          id,
          title: "Cuộc trò chuyện mới",
          messages: [],
          createdAt: new Date().toISOString(),
        },
      ]);
      setActiveId(id);
    }
  }, [conversations.length]);

  const newConversation = () => {
    const id = Date.now().toString();
    const conv = {
      id,
      title: "Cuộc trò chuyện mới",
      messages: [],
      createdAt: new Date().toISOString(),
    };
    setConversations((s) => [conv, ...s]);
    setActiveId(id);
  };

  const sendMessage = async (text) => {
    if (!text || !text.trim()) return;
    if (!activeId) return;

    const conversationId = activeId;
    const assistantMsgId = `${Date.now()}-assistant`;
    const userMsg = { role: "user", content: text, createdAt: new Date().toISOString() };

    setConversations((prev) =>
      prev.map((c) => (c.id === conversationId ? { ...c, messages: [...c.messages, userMsg] } : c))
    );

    setLoading(true);
    try {
      const active = conversations.find((c) => c.id === conversationId) || { messages: [] };

      setConversations((prev) =>
        prev.map((c) =>
          c.id === conversationId
            ? {
                ...c,
                messages: [
                  ...c.messages,
                  {
                    id: assistantMsgId,
                    role: "assistant",
                    content: "",
                    createdAt: new Date().toISOString(),
                  },
                ],
              }
            : c
        )
      );

      const apiBase = import.meta.env.VITE_API_URL || "http://localhost:8000";
      const payload = { case_description: text };
      const res = await fetch(`${apiBase}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(await res.text());
      }

      const data = await res.json();
      const assistantContent = data.analysis || "Phân tích không có nội dung";

      setConversations((prev) =>
        prev.map((c) =>
          c.id === conversationId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === assistantMsgId
                    ? { ...m, content: assistantContent }
                    : m
                ),
              }
            : c
        )
      );
    } catch (err) {
      console.error(err);
      setConversations((prev) =>
        prev.map((c) =>
          c.id === conversationId
            ? {
                ...c,
                messages: c.messages.map((m) =>
                  m.id === assistantMsgId && m.content === ""
                    ? {
                        ...m,
                        content: "Lỗi: Không thể kết nối tới backend.",
                      }
                    : m
                ),
              }
            : c
        )
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: "flex", height: "100svh", bgcolor: "background.default" }}>
        <Sidebar
          conversations={conversations}
          activeId={activeId}
          setActiveId={setActiveId}
          newConversation={newConversation}
        />

        <Box sx={{ flex: 1, display: "flex", flexDirection: "column" }}>
          <ChatWindow conversation={activeConversation} />

          <Box sx={{ px: 2, py: 1, borderTop: 1, borderColor: "divider", bgcolor: "background.paper" }}>
            <ChatInput onSend={sendMessage} disabled={loading} />
            {loading && <LoadingIndicator />}
          </Box>
        </Box>
      </Box>
    </ThemeProvider>
  );
}
