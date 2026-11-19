// AmevaSdPlainHarness: the plain txt2img half of AmevaSdRefHarness, using only API that exists before and
// after the reference-image change, so the same class runs against the old SDK classes (AmfyUI.apk dex) and
// the new ones (AAR classes.jar). Run with app_process as described in AmevaSdRefHarness.
import android.content.Context;
import android.content.ContextWrapper;
import com.ameva.amuda.diffusion.AmevaDiffusion;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.PrintStream;

public class AmevaSdPlainHarness {
    static long fnv(byte[] b) {
        long h = 0xcbf29ce484222325L;
        for (byte x : b) { h ^= (x & 0xFF); h *= 0x100000001b3L; }
        return h;
    }

    public static void main(String[] args) {
        PrintStream out = new PrintStream(new FileOutputStream(FileDescriptor.out), true);
        try {
            Context ctx = new ContextWrapper(null);
            AmevaDiffusion.LoadOptions o = new AmevaDiffusion.LoadOptions();
            o.setModelPath(args[0]);
            o.setCacheDir(args[1]);
            o.setGpuBudgetMode(AmevaDiffusion.GpuBudgetMode.CPU_ONLY);
            o.setNThreads(4);
            AmevaDiffusion d = AmevaDiffusion.load(ctx, o);
            long[] seeds = {42L, 7L};
            for (long s : seeds) {
                AmevaDiffusion.GenerateParams p = new AmevaDiffusion.GenerateParams();
                p.setPrompt("a red apple on a wooden table");
                p.setWidth(128);
                p.setHeight(128);
                p.setSteps(1);
                p.setCfgScale(1.0f);
                p.setSeed(s);
                byte[] img = d.generate(p, null);
                out.println("plain seed=" + s + " bytes=" + img.length + String.format(" fnv=%016x", fnv(img)));
            }
            d.close();
        } catch (Throwable t) {
            out.println("EXCEPTION " + t);
            t.printStackTrace(out);
        }
        out.flush();
        System.exit(0);
    }
}
