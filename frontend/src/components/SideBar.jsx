import React from "react";
import { Box, List, ListItemButton, ListItemText, Typography, Divider, Button, Avatar } from "@mui/material";
import AddIcon from "@mui/icons-material/Add";

export default function Sidebar({ conversations = [], activeId, setActiveId, newConversation }) {
	return (
		<Box sx={{ width: { xs: 80, sm: 320 }, borderRight: 1, borderColor: "divider", bgcolor: "background.paper", display: "flex", flexDirection: "column" }}>
			<Box sx={{ p: 2, display: "flex", alignItems: "center", gap: 2 }}>
				<Avatar sx={{ bgcolor: "primary.main" }}>AL</Avatar>
				<Box sx={{ display: { xs: "none", sm: "block" } }}>
					<Typography variant="h6">AI Lawyer</Typography>
					<Typography variant="caption" color="text.secondary">Trợ lý pháp lý</Typography>
				</Box>
			</Box>

			<Divider />

			<Box sx={{ p: 1 }}>
				<Button startIcon={<AddIcon />} fullWidth variant="outlined" onClick={newConversation}>
					Cuộc trò chuyện mới
				</Button>
			</Box>

			<Divider />

			<Box sx={{ overflow: "auto", flex: 1 }}>
				<List>
					{conversations.map((c) => (
						<ListItemButton key={c.id} selected={c.id === activeId} onClick={() => setActiveId(c.id)} sx={{ '&:hover': { bgcolor: 'action.hover' } }}>
							<ListItemText primary={c.title} secondary={new Date(c.createdAt).toLocaleString()} />
						</ListItemButton>
					))}
				</List>
			</Box>

			<Box sx={{ p: 2 }}>
				<Typography variant="caption" color="text.secondary">Phiên bản thử nghiệm • Không dùng cho tư vấn pháp lý chính thức</Typography>
			</Box>
		</Box>
	);
}
