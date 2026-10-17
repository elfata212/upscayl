import { useEffect, useState } from "react";

const useSystemInfo = () => {
  const [systemInfo, setSystemInfo] = useState<Awaited<
    ReturnType<typeof window.electron.getSystemInfo>
  > | null>(null);

  useEffect(() => {
    const electronApi = typeof window !== "undefined" ? (window as any).electron : undefined;
    if (!electronApi?.getSystemInfo) return;
    const getSystemInfo = async () => {
      try {
        const info = await electronApi.getSystemInfo();
        setSystemInfo(info);
      } catch (error) {
        console.warn("Unable to read desktop system information:", error);
      }
    };
    void getSystemInfo();
  }, []);
  return { systemInfo };
};

export default useSystemInfo;
