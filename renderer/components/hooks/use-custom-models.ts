import { ELECTRON_COMMANDS } from "@common/electron-commands";
import { useEffect } from "react";
import useLogger from "./use-logger";

export const initCustomModels = () => {
  const logit = useLogger();

  useEffect(() => {
    const electronApi = typeof window !== "undefined" ? (window as any).electron : undefined;
    if (!electronApi?.send) return;
    let customModelsPath: unknown = null;
    try {
      customModelsPath = JSON.parse(localStorage.getItem("customModelsPath") ?? "null");
    } catch {
      customModelsPath = null;
    }
    if (customModelsPath !== null) {
      electronApi.send(ELECTRON_COMMANDS.GET_MODELS_LIST, customModelsPath);
      logit("🎯 GET_MODELS_LIST: ", customModelsPath);
    }
  }, []);
};
