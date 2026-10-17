"use client";

import { useEffect, useRef, useState } from "react";

const MODEL_URL =
  "https://huggingface.co/skillsafe-ai/realesrgan-x4plus/resolve/main/model.onnx";
const ORT_VERSION = "1.30.0";
const MAX_INPUT_SIDE = 768;
const TILE_SIZE = 64;
const OVERLAP = 8;

type PickedImage = { file: File; url: string };

async function loadOnnxRuntime(): Promise<any> {
  const browserWindow = window as any;
  if (browserWindow.ort) return browserWindow.ort;

  await new Promise<void>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>(
      'script[data-upscayl-ort="true"]',
    );
    if (existing) {
      if (browserWindow.ort) {
        resolve();
        return;
      }
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener(
        "error",
        () => reject(new Error("Runtime AI gagal dimuat dari CDN.")),
        { once: true },
      );
      return;
    }

    const script = document.createElement("script");
    script.dataset.upscaylOrt = "true";
    script.src = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.min.js`;
    script.async = true;
    script.onload = () => resolve();
    script.onerror = () =>
      reject(new Error("Tidak dapat memuat runtime AI. Periksa koneksi internet."));
    document.head.appendChild(script);
  });

  if (!browserWindow.ort) {
    throw new Error("Runtime AI termuat tetapi tidak dapat diinisialisasi.");
  }
  browserWindow.ort.env.wasm.wasmPaths =
    `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
  browserWindow.ort.env.wasm.numThreads = 1;
  return browserWindow.ort;
}

async function loadModel(ort: any, onStatus: (text: string) => void): Promise<any> {
  const browserWindow = window as any;
  if (browserWindow.__upscaylWebSession) return browserWindow.__upscaylWebSession;
  if (browserWindow.__upscaylWebSessionPromise) {
    return browserWindow.__upscaylWebSessionPromise;
  }

  onStatus("Mengunduh model AI Real-ESRGAN (sekitar 64 MB, pertama kali saja)…");
  browserWindow.__upscaylWebSessionPromise = ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  })
    .then((session: any) => {
      browserWindow.__upscaylWebSession = session;
      return session;
    })
    .catch((error: unknown) => {
      browserWindow.__upscaylWebSessionPromise = null;
      throw error;
    });
  return browserWindow.__upscaylWebSessionPromise;
}

async function runUpscale(
  file: File,
  onStatus: (text: string) => void,
  onProgress: (percent: number) => void,
): Promise<Blob> {
  const ort = await loadOnnxRuntime();
  const session = await loadModel(ort, onStatus);

  onStatus("Membaca dan menyiapkan gambar…");
  const bitmap = await createImageBitmap(file);
  const shrink = Math.min(1, MAX_INPUT_SIDE / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * shrink));
  const height = Math.max(1, Math.round(bitmap.height * shrink));

  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  if (!sourceContext) throw new Error("Browser tidak dapat membaca gambar.");
  sourceContext.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  const scale = 4;
  const outputWidth = width * scale;
  const outputHeight = height * scale;
  const outputCanvas = document.createElement("canvas");
  outputCanvas.width = outputWidth;
  outputCanvas.height = outputHeight;
  const outputContext = outputCanvas.getContext("2d", { willReadFrequently: true });
  if (!outputContext) throw new Error("Browser tidak dapat membuat gambar hasil.");

  const finalImage = outputContext.createImageData(outputWidth, outputHeight);
  const columns = Math.ceil(width / TILE_SIZE);
  const rows = Math.ceil(height / TILE_SIZE);
  const totalTiles = columns * rows;
  let completedTiles = 0;

  for (let y = 0; y < height; y += TILE_SIZE) {
    for (let x = 0; x < width; x += TILE_SIZE) {
      const coreWidth = Math.min(TILE_SIZE, width - x);
      const coreHeight = Math.min(TILE_SIZE, height - y);
      const cropX = Math.max(0, x - OVERLAP);
      const cropY = Math.max(0, y - OVERLAP);
      const cropRight = Math.min(width, x + coreWidth + OVERLAP);
      const cropBottom = Math.min(height, y + coreHeight + OVERLAP);
      const cropWidth = cropRight - cropX;
      const cropHeight = cropBottom - cropY;
      const pixels = sourceContext.getImageData(cropX, cropY, cropWidth, cropHeight).data;
      const tensorData = new Float32Array(3 * cropWidth * cropHeight);
      for (let py = 0; py < cropHeight; py++) {
        for (let px = 0; px < cropWidth; px++) {
          const pixel = (py * cropWidth + px) * 4;
          const planar = py * cropWidth + px;
          tensorData[planar] = pixels[pixel] / 255;
          tensorData[cropWidth * cropHeight + planar] = pixels[pixel + 1] / 255;
          tensorData[2 * cropWidth * cropHeight + planar] = pixels[pixel + 2] / 255;
        }
      }

      const feedName = session.inputNames[0] || "input";
      const outputName = session.outputNames[0] || "output";
      const result = await session.run({
        [feedName]: new ort.Tensor("float32", tensorData, [1, 3, cropHeight, cropWidth]),
      });
      const output = result[outputName] || result[session.outputNames[0]];
      const outputData = output.data as Float32Array;
      const inferredWidth = cropWidth * scale;
      const offsetX = (x - cropX) * scale;
      const offsetY = (y - cropY) * scale;

      for (let py = 0; py < coreHeight * scale; py++) {
        const srcRow = (offsetY + py) * inferredWidth;
        const dstRow = (y * scale + py) * outputWidth + x * scale;
        for (let px = 0; px < coreWidth * scale; px++) {
          const src = (srcRow + offsetX + px) * 3;
          const dst = (dstRow + px) * 4;
          finalImage.data[dst] = Math.max(0, Math.min(255, Math.round(outputData[src] * 255)));
          finalImage.data[dst + 1] = Math.max(0, Math.min(255, Math.round(outputData[src + 1] * 255)));
          finalImage.data[dst + 2] = Math.max(0, Math.min(255, Math.round(outputData[src + 2] * 255)));
          finalImage.data[dst + 3] = 255;
        }
      }

      completedTiles++;
      onProgress(Math.round((completedTiles / totalTiles) * 100));
      onStatus(`AI upscaling berjalan: tile ${completedTiles} dari ${totalTiles}`);
      // Yield to the browser so progress updates can paint between tiles.
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    }
  }

  // Preserve transparency when the source image has an alpha channel.
  const alphaCanvas = document.createElement("canvas");
  alphaCanvas.width = outputWidth;
  alphaCanvas.height = outputHeight;
  const alphaContext = alphaCanvas.getContext("2d", { willReadFrequently: true });
  if (alphaContext) {
    alphaContext.imageSmoothingEnabled = true;
    alphaContext.imageSmoothingQuality = "high";
    alphaContext.drawImage(sourceCanvas, 0, 0, outputWidth, outputHeight);
    const alphaData = alphaContext.getImageData(0, 0, outputWidth, outputHeight).data;
    for (let i = 0; i < finalImage.data.length; i += 4) {
      finalImage.data[i + 3] = alphaData[i + 3];
    }
  }

  outputContext.putImageData(finalImage, 0, 0);
  onStatus("Menyimpan hasil AI…");
  return await new Promise<Blob>((resolve, reject) => {
    outputCanvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error("Gagal membuat file PNG hasil."));
    }, "image/png");
  });
}

export default function WebUpscaler() {
  const [picked, setPicked] = useState<PickedImage | null>(null);
  const [resultUrl, setResultUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Pilih gambar untuk memulai.");
  const [error, setError] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);
  const dragCounter = useRef(0);

  useEffect(() => () => {
    if (picked?.url) URL.revokeObjectURL(picked.url);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
  }, [picked?.url, resultUrl]);

  const chooseFile = (file?: File) => {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("File tidak dikenali sebagai gambar.");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("Untuk menjaga kestabilan browser Android, pilih gambar maksimal 20 MB.");
      return;
    }
    setError("");
    setResultUrl("");
    setProgress(0);
    setStatus("Gambar siap diproses.");
    setPicked({ file, url: URL.createObjectURL(file) });
  };

  const startUpscale = async () => {
    if (!picked || busy) return;
    setBusy(true);
    setError("");
    setProgress(0);
    try {
      const blob = await runUpscale(picked.file, setStatus, setProgress);
      setResultUrl(URL.createObjectURL(blob));
      setProgress(100);
      setStatus("Selesai. Hasil AI 4× siap diunduh.");
    } catch (err) {
      console.error("Upscayl web inference failed:", err);
      setError(err instanceof Error ? err.message : "Proses AI gagal. Coba gambar yang lebih kecil atau muat ulang halaman.");
      setStatus("Proses tidak selesai.");
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    if (busy) return;
    setPicked(null);
    setResultUrl("");
    setProgress(0);
    setError("");
    setStatus("Pilih gambar untuk memulai.");
    if (fileInput.current) fileInput.current.value = "";
  };

  return (
    <main className="min-h-screen bg-slate-950 px-4 py-8 text-slate-100 sm:px-8">
      <div className="mx-auto max-w-5xl">
        <header className="mb-8 flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.25em] text-emerald-400">Upscayl Web</p>
            <h1 className="mt-2 text-3xl font-bold tracking-tight sm:text-4xl">AI Image Upscaler</h1>
            <p className="mt-2 max-w-xl text-sm text-slate-400">Perbesar gambar hingga 4× langsung di browser Android atau komputer.</p>
          </div>
          <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1.5 text-xs font-medium text-emerald-300">Pemrosesan lokal</span>
        </header>

        <div className="grid gap-6 lg:grid-cols-[1fr_0.85fr]">
          <section
            className="rounded-2xl border border-slate-800 bg-slate-900 p-4 sm:p-6"
            onDragEnter={(event) => { event.preventDefault(); dragCounter.current++; }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={(event) => { event.preventDefault(); dragCounter.current = Math.max(0, dragCounter.current - 1); }}
            onDrop={(event) => { event.preventDefault(); dragCounter.current = 0; chooseFile(event.dataTransfer.files?.[0]); }}
          >
            <div className="mb-4 flex items-center justify-between gap-3">
              <h2 className="text-lg font-semibold">Gambar sumber</h2>
              {picked && <button type="button" onClick={reset} disabled={busy} className="text-sm text-slate-400 underline disabled:opacity-50">Ganti gambar</button>}
            </div>
            {picked ? (
              <div className="flex min-h-64 items-center justify-center rounded-xl bg-slate-950 p-3">
                <img src={picked.url} alt="Pratinjau gambar sumber" className="max-h-[420px] max-w-full rounded-lg object-contain" />
              </div>
            ) : (
              <button type="button" onClick={() => fileInput.current?.click()} className="flex min-h-64 w-full flex-col items-center justify-center rounded-xl border-2 border-dashed border-slate-700 bg-slate-950/60 px-4 text-center transition hover:border-emerald-400">
                <span className="mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-slate-800 text-2xl">＋</span>
                <span className="font-semibold">Pilih atau jatuhkan gambar di sini</span>
                <span className="mt-2 text-sm text-slate-400">JPG, PNG, WebP • maksimal 20 MB</span>
              </button>
            )}
            <input ref={fileInput} type="file" accept="image/*" className="hidden" onChange={(event) => chooseFile(event.target.files?.[0])} />
            {picked && (
              <p className="mt-3 break-words text-xs text-slate-400">{picked.file.name} · {(picked.file.size / (1024 * 1024)).toFixed(2)} MB</p>
            )}
            <button type="button" onClick={startUpscale} disabled={!picked || busy} className="mt-5 w-full rounded-xl bg-emerald-400 px-4 py-3.5 font-bold text-slate-950 transition hover:bg-emerald-300 disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400">
              {busy ? "Sedang memproses AI…" : "Upscale dengan AI · 4×"}
            </button>
            {busy && (
              <div className="mt-4">
                <div className="mb-2 flex justify-between gap-3 text-xs text-slate-300"><span>{status}</span><span>{progress}%</span></div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-800"><div className="h-full rounded-full bg-emerald-400 transition-all" style={{ width: `${progress}%` }} /></div>
              </div>
            )}
            {error && <p role="alert" className="mt-4 rounded-xl border border-red-500/30 bg-red-950/50 p-3 text-sm text-red-200">{error}</p>}
            {!busy && <p className="mt-4 text-xs leading-5 text-slate-500">{status}</p>}
          </section>

          <section className="rounded-2xl border border-slate-800 bg-slate-900 p-4 sm:p-6">
            <h2 className="mb-4 text-lg font-semibold">Hasil upscaling</h2>
            {resultUrl ? (
              <>
                <div className="flex min-h-64 items-center justify-center rounded-xl bg-[linear-gradient(45deg,#1e293b_25%,transparent_25%),linear-gradient(-45deg,#1e293b_25%,transparent_25%),linear-gradient(45deg,transparent_75%,#1e293b_75%),linear-gradient(-45deg,transparent_75%,#1e293b_75%)] bg-[length:20px_20px] bg-[position:0_0,0_10px,10px_-10px,-10px_0px] p-3">
                  <img src={resultUrl} alt="Hasil AI upscaling" className="max-h-[420px] max-w-full rounded-lg object-contain" />
                </div>
                <a href={resultUrl} download={`${picked?.file.name.replace(/\.[^.]+$/, "") || "image"}_upscayl_4x.png`} className="mt-5 block rounded-xl bg-white px-4 py-3.5 text-center font-bold text-slate-950 transition hover:bg-slate-200">Unduh hasil PNG</a>
                <p className="mt-3 text-xs leading-5 text-slate-500">Hasil diperbesar 4× dan dibuat sebagai file PNG.</p>
              </>
            ) : (
              <div className="flex min-h-64 flex-col items-center justify-center rounded-xl border border-slate-800 bg-slate-950/50 px-5 text-center">
                <span className="text-4xl text-slate-600">✧</span>
                <p className="mt-3 font-medium text-slate-300">Hasil akan muncul di sini</p>
                <p className="mt-2 max-w-xs text-sm leading-6 text-slate-500">Model Real-ESRGAN berjalan pada perangkat Anda. Gambar tidak diunggah ke server aplikasi.</p>
              </div>
            )}
          </section>
        </div>

        <footer className="mt-6 space-y-2 text-xs leading-5 text-slate-500">
          <p>Model AI pertama kali memerlukan unduhan sekitar 64 MB. Waktu pemrosesan bergantung pada perangkat. Untuk kestabilan browser ponsel, sisi terpanjang gambar masukan dibatasi 768 piksel pada tahap AI.</p>
          <p>Model Real-ESRGAN x4plus didistribusikan dengan lisensi BSD-3-Clause. Upscayl Web ini adalah jalur browser terpisah dari mesin Electron/NCNN pada aplikasi desktop.</p>
        </footer>
      </div>
    </main>
  );
}
