import React from "react";
import { Box, CircularProgress, Typography } from "@mui/material";

export default function LoadingIndicator() {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 2, mt: 1 }}>
      <CircularProgress size={20} />
      <Typography variant="body2" color="text.secondary">AI đang phân tích...</Typography>
    </Box>
  );
}
