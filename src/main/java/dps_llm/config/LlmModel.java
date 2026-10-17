package dps_llm.config;

import java.util.Locale;

/**
 * The OpenRouter models the LLM pipeline can run, chosen with a command-line flag.
 * <p>
 * CHANGE: replaces the single {@code OPENROUTER_MODEL} key. Each model has its own .env key
 * holding the OpenRouter model id, and its own default output CSV so that a run with one model
 * never overwrites the summaries of another (or the canonical Mixtral run in
 * {@code llm_summaries.csv}).
 * </p>
 * <pre>
 * mvn exec:java@llm-gpt        (same as passing --gpt)
 * mvn exec:java@llm-mistral    (same as passing --mistral)
 * </pre>
 */
public enum LlmModel {

    GPT("gpt", "GPT_MODEL"),
    MISTRAL("mistral", "MISTRAL_MODEL");

    private final String flag;
    private final String envKey;

    LlmModel(String flag, String envKey) {
        this.flag = flag;
        this.envKey = envKey;
    }

    /** Flag name without dashes, e.g. {@code gpt}. */
    public String flag() {
        return flag;
    }

    /** The .env key that holds this model's OpenRouter id. */
    public String envKey() {
        return envKey;
    }

    /** Default summary CSV for this model, e.g. {@code output/summary-output/llm_summaries_gpt.csv}. */
    public String defaultOutputPath() {
        return "output/summary-output/llm_summaries_" + flag + ".csv";
    }

    /**
     * Picks the model from the command-line arguments. Exactly one of {@code --gpt} or
     * {@code --mistral} is required; single-dash forms ({@code -gpt}) and any letter case are
     * accepted.
     *
     * @param args command-line arguments
     * @return the selected model
     * @throws IllegalArgumentException if no model, more than one, or an unknown argument is given
     */
    public static LlmModel fromArgs(String[] args) {
        LlmModel selected = null;
        for (String arg : args == null ? new String[0] : args) {
            if (arg == null || arg.isBlank()) {
                continue;
            }
            String name = arg.trim().replaceFirst("^-{1,2}", "").toLowerCase(Locale.ROOT);
            LlmModel match = null;
            for (LlmModel model : values()) {
                if (model.flag.equals(name)) {
                    match = model;
                }
            }
            if (match == null) {
                throw new IllegalArgumentException("Unknown argument '" + arg + "'.");
            }
            if (selected != null && selected != match) {
                throw new IllegalArgumentException("Choose one model: --" + selected.flag + " or --" + match.flag + ", not both.");
            }
            selected = match;
        }
        if (selected == null) {
            throw new IllegalArgumentException("No model selected.");
        }
        return selected;
    }
}
