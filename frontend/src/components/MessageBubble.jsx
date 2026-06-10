import React from "react";
import { Box, Paper, Typography } from "@mui/material";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

function timeLabel(ts) {
  try {
    return new Date(ts).toLocaleString();
  } catch {
    return "";
  }
}

export default function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <Box sx={{ display: "flex", flexDirection: isUser ? "row-reverse" : "row", mb: 1 }}>
      <Paper elevation={0} sx={{ maxWidth: "80%", p: 1.5, bgcolor: isUser ? "primary.dark" : "background.paper", color: isUser ? "text.primary" : "text.primary", borderRadius: 2 }}>
        <Box sx={{ whiteSpace: "pre-wrap" }}>
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={{
            code({node, inline, className, children, ...props}){
              return <Box component="code" sx={{ display: 'block', p:1, bgcolor: 'background.default', borderRadius:1, fontFamily:'monospace' }} {...props}>{children}</Box>
            }
          }}>{message.content}</ReactMarkdown>
        </Box>
        <Typography variant="caption" sx={{ display: 'block', mt: 0.5, color: 'text.secondary', textAlign: isUser ? 'right' : 'left' }}>{timeLabel(message.createdAt)}</Typography>
      </Paper>
    </Box>
  );
}
