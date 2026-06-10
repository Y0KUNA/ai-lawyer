import { createTheme } from "@mui/material/styles";

const theme = createTheme({
  palette: {
    mode: "dark",
    primary: { main: "#10b981" },
    background: { default: "#0b1220", paper: "#071224" },
    divider: "rgba(255,255,255,0.08)",
  },
  typography: {
    fontFamily: "Inter, Roboto, Arial, sans-serif",
    h6: { fontWeight: 600 },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: { backgroundImage: "none" },
      },
    },
  },
});

export default theme;
