import React from "react";
import { createRoot } from "react-dom/client";
import VitaTwinApp from "../frontend_App.jsx";

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <VitaTwinApp />
  </React.StrictMode>,
);
