package dps_llm.config;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Command-line options of the LLM pipeline: the model ({@code --gpt} / {@code --mistral}) and,
 * optionally, {@code --nonconcise} to use the non-concise 50-word prompt instead of the concise one.
 * <p>
 * CHANGE: added so the non-concise prompt can be run per model without editing code. Each
 * model/prompt combination writes its own CSV:
 * </p>
 * <pre>
 * --gpt                  output/summary-output/llm_summaries_gpt.csv
 * --mistral              output/summary-output/llm_summaries_mistral.csv
 * --gpt --nonconcise     output/summary-output/llm_summaries_nonconcise_gpt.csv
 * --mistral --nonconcise output/summary-output/llm_summaries_nonconcise_mistral.csv
 * --gpt + SENIOR_ANALYST_20_WORDS  output/summary-output/llm_summaries_gpt_20.csv (also 40, 60, 80)
 * </pre>
 * <p>
 * CHANGE: {@code --resume} keeps the rows already in that CSV and only requests the classes that
 * are missing, e.g. after a run stopped part-way because of rate limits.
 * </p>
 */
public final class LlmRunOptions {

    /** Prompt used by a normal run (prompts.json). */
    public static final String CONCISE_PROMPT_ALIAS = "SENIOR_ANALYST_50_WORDS";
    /** Prompt used with --nonconcise (prompts.json). */
    public static final String NON_CONCISE_PROMPT_ALIAS = "SENIOR_ANALYST_50_WORDS_NON_CONCISE";

    /** Printed when the arguments are missing or invalid. */
    public static final String USAGE = "Usage: DpsLlmApplication --gpt | --mistral [--nonconcise] [--resume]\n"
            + "  --gpt         use the model id in GPT_MODEL (.env)\n"
            + "  --mistral     use the model id in MISTRAL_MODEL (.env)\n"
            + "  --nonconcise  use the non-concise 50-word prompt instead of the concise one\n"
            + "  --resume      keep the summaries already in the output CSV and only generate the missing ones";

    private static final String SUMMARY_DIR = "output/summary-output/";
    /** CHANGE: the length-variant prompts in prompts.json, e.g. SENIOR_ANALYST_20_WORDS. */
    private static final Pattern LENGTH_PROMPT_ALIAS = Pattern.compile("SENIOR_ANALYST_(\\d+)_WORDS");

    private final LlmModel model;
    private final boolean nonConcise;
    private final boolean resume;

    LlmRunOptions(LlmModel model, boolean nonConcise) {
        this(model, nonConcise, false);
    }

    LlmRunOptions(LlmModel model, boolean nonConcise, boolean resume) {
        this.model = model;
        this.nonConcise = nonConcise;
        this.resume = resume;
    }

    /**
     * Parses the command-line arguments. Exactly one model flag is required; {@code --nonconcise}
     * (also {@code --non-concise}) and {@code --resume} are optional. Single dashes and any letter case are accepted.
     *
     * @param args command-line arguments
     * @return the parsed options
     * @throws IllegalArgumentException if the model flag is missing, repeated with another model, or unknown
     */
    public static LlmRunOptions parse(String[] args) {
        boolean nonConcise = false;
        boolean resume = false;
        List<String> modelArgs = new ArrayList<>();
        for (String arg : args == null ? new String[0] : args) {
            if (arg == null || arg.isBlank()) {
                continue;
            }
            String name = arg.trim().replaceFirst("^-{1,2}", "").toLowerCase(Locale.ROOT);
            if (name.equals("nonconcise") || name.equals("non-concise")) {
                nonConcise = true;
            } else if (name.equals("resume")) {
                resume = true;
            } else {
                modelArgs.add(arg);
            }
        }
        // return new LlmRunOptions(LlmModel.fromArgs(modelArgs.toArray(new String[0])), nonConcise);
        return new LlmRunOptions(LlmModel.fromArgs(modelArgs.toArray(new String[0])), nonConcise, resume);
    }

    public LlmModel model() {
        return model;
    }

    public boolean nonConcise() {
        return nonConcise;
    }

    /** True when the output CSV is kept and only missing classes are summarised. */
    public boolean resume() {
        return resume;
    }

    /** The prompts.json alias this run uses. */
    public String promptAlias() {
        return nonConcise ? NON_CONCISE_PROMPT_ALIAS : CONCISE_PROMPT_ALIAS;
    }

    /** Default summary CSV for this run, e.g. {@code llm_summaries_nonconcise_gpt.csv}. */
    public String defaultOutputPath() {
        return outputPathFor(promptAlias());
    }

    /**
     * Default summary CSV for any prompt alias with this run's model. The concise and non-concise
     * prompts get short names; other aliases (from LLM_PROMPT_ALIASES) use the alias itself, e.g.
     * {@code llm_summaries_senior_analyst_20_words_gpt.csv}.
     * CHANGE: the length variants SENIOR_ANALYST_&lt;n&gt;_WORDS now follow the normal run's name with the
     * length appended, e.g. {@code llm_summaries_gpt_20.csv}.
     */
    public String outputPathFor(String alias) {
        if (alias == null || alias.isBlank() || CONCISE_PROMPT_ALIAS.equals(alias)) {
            return model.defaultOutputPath();
        }
        if (NON_CONCISE_PROMPT_ALIAS.equals(alias)) {
            return SUMMARY_DIR + "llm_summaries_nonconcise_" + model.flag() + ".csv";
        }
        // CHANGE: SENIOR_ANALYST_20_WORDS used to map to llm_summaries_senior_analyst_20_words_<model>.csv.
        Matcher length = LENGTH_PROMPT_ALIAS.matcher(alias);
        if (length.matches()) {
            return SUMMARY_DIR + "llm_summaries_" + model.flag() + "_" + length.group(1) + ".csv";
        }
        String sanitized = alias.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "_").replaceAll("^_+|_+$", "");
        if (sanitized.isEmpty()) {
            sanitized = "custom";
        }
        return SUMMARY_DIR + "llm_summaries_" + sanitized + "_" + model.flag() + ".csv";
    }
}
