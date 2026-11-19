// AmevaSdRefHarness: Kotlin/JNI path of the reference-image fields, run on a phone with app_process
// (no app, no adb). CPU_ONLY with a real single-file model; no Context is dereferenced on that path.
//
//   app_process -cp harness.jar /system/bin AmevaSdRefHarness <model.gguf> <cacheDir>
//
// Expects: a plain txt2img gives the same bytes as the C driver (tests/ameva_sd_ref_device.cpp, same
// prompt/size/steps/cfg/seed); every malformed or unsupported reference-image request throws AmudaException
// with code 5 (AMEVA_SD_ERR_INVALID_ARGUMENT) and the next plain generation is unchanged.
import android.content.Context;
import android.content.ContextWrapper;
import com.ameva.amuda.AmudaException;
import com.ameva.amuda.diffusion.AmevaDiffusion;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;

public class AmevaSdRefHarness {
    static PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out), true);
    static int bad = 0;

    static long fnv(byte[] b) {
        long h = 0xcbf29ce484222325L;
        for (byte x : b) { h ^= (x & 0xFF); h *= 0x100000001b3L; }
        return h;
    }

    static AmevaDiffusion.GenerateParams base() {
        AmevaDiffusion.GenerateParams p = new AmevaDiffusion.GenerateParams();
        p.setPrompt("a red apple on a wooden table");
        p.setWidth(128);
        p.setHeight(128);
        p.setSteps(1);
        p.setCfgScale(1.0f);
        p.setSeed(42L);
        return p;
    }

    static void plain(AmevaDiffusion d, String label) {
        byte[] img = d.generate(base(), null);
        out.println(label + " bytes=" + img.length + String.format(" fnv=%016x", fnv(img)));
    }

    static void refused(AmevaDiffusion d, String label, AmevaDiffusion.GenerateParams p, String needle) {
        try {
            d.generate(p, null);
            out.println("FAIL " + label + ": no exception");
            bad++;
        } catch (AmudaException e) {
            boolean ok = e.getCode() == 5 && e.getMessage() != null && e.getMessage().contains(needle);
            out.println((ok ? "ok   " : "FAIL ") + label + ": code=" + e.getCode() + " msg=" + e.getMessage());
            if (!ok) bad++;
        } catch (Throwable t) {
            out.println("FAIL " + label + ": " + t);
            bad++;
        }
    }

    public static void main(String[] args) {
        try {
            Context ctx = new ContextWrapper(null); // never dereferenced: CPU_ONLY and cacheDir are set
            AmevaDiffusion.LoadOptions o = new AmevaDiffusion.LoadOptions();
            o.setModelPath(args[0]);
            o.setCacheDir(args[1]);
            o.setGpuBudgetMode(AmevaDiffusion.GpuBudgetMode.CPU_ONLY);
            o.setNThreads(4);
            AmevaDiffusion d = AmevaDiffusion.load(ctx, o);

            plain(d, "plain txt2img seed=42");

            byte[] img = new byte[8 * 8 * 3];
            AmevaDiffusion.GenerateParams p = base();
            p.setRefImages(new byte[][] {img});
            p.setRefWidths(new int[] {8});
            p.setRefHeights(new int[] {8});
            refused(d, "1 ref on single-file checkpoint", p, "diffusion_model_path");

            p = base();
            p.setRefImages(new byte[][] {img, img, img, img});
            p.setRefWidths(new int[] {8, 8, 8, 8});
            p.setRefHeights(new int[] {8, 8, 8, 8});
            refused(d, "4 refs", p, "ref_count");

            p = base();
            p.setRefImages(new byte[][] {img});
            p.setRefWidths(new int[] {8, 8});
            p.setRefHeights(new int[] {8});
            refused(d, "widths length mismatch", p, "refImages");

            p = base();
            p.setRefImages(new byte[][] {new byte[10]});
            p.setRefWidths(new int[] {8});
            p.setRefHeights(new int[] {8});
            // The single-file model gate runs before the per-image checks (those are covered, with a multi-file
            // model, by tests/ameva_sd_ref_check.cpp); here the exception mapping and code are what is checked.
            refused(d, "buffer size mismatch (model gate first)", p, "diffusion_model_path");

            p = base();
            p.setRefImages(new byte[][] {null});
            p.setRefWidths(new int[] {8});
            p.setRefHeights(new int[] {8});
            refused(d, "null image (model gate first)", p, "diffusion_model_path");

            p = base();
            p.setRefImageArgs("preset=qwen");
            refused(d, "args without images", p, "without reference");

            plain(d, "plain txt2img seed=42 after refusals");
            d.close();
            out.println(bad == 0 ? "result OK" : "result BAD (" + bad + ")");
        } catch (Throwable t) {
            out.println("EXCEPTION " + t);
            t.printStackTrace(out);
            out.println("result BAD");
        }
        out.flush();
        System.exit(bad == 0 ? 0 : 1);
    }
}
