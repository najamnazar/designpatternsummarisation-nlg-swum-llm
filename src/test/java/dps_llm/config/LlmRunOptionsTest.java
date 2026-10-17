package dps_llm.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Tests the --gpt / --mistral [--nonconcise] options and the CSV each run writes. */
class LlmRunOptionsTest {

    @Test
    void conciseIsTheDefaultPrompt() {
        LlmRunOptions options = LlmRunOptions.parse(new String[] {"--gpt"});
        assertEquals(LlmModel.GPT, options.model());
        assertFalse(options.nonConcise());
        assertFalse(options.resume());
        assertEquals("SENIOR_ANALYST_50_WORDS", options.promptAlias());
        assertEquals("output/summary-output/llm_summaries_gpt.csv", options.defaultOutputPath());
    }

    @Test
    void nonConciseFlagSwitchesPromptAndOutputInAnyOrder() {
        LlmRunOptions gpt = LlmRunOptions.parse(new String[] {"--nonconcise", "--gpt"});
        LlmRunOptions mistral = LlmRunOptions.parse(new String[] {"-Mistral", "--non-concise"});
        assertTrue(gpt.nonConcise());
        assertEquals("SENIOR_ANALYST_50_WORDS_NON_CONCISE", gpt.promptAlias());
        assertEquals("output/summary-output/llm_summaries_nonconcise_gpt.csv", gpt.defaultOutputPath());
        assertEquals(LlmModel.MISTRAL, mistral.model());
        assertEquals("output/summary-output/llm_summaries_nonconcise_mistral.csv", mistral.defaultOutputPath());
    }

    @Test
    void everyModelAndPromptCombinationHasItsOwnFile() {
        LlmRunOptions options = LlmRunOptions.parse(new String[] {"--gpt"});
        assertEquals("output/summary-output/llm_summaries_nonconcise_gpt.csv",
                options.outputPathFor("SENIOR_ANALYST_50_WORDS_NON_CONCISE"));
        // assertEquals("output/summary-output/llm_summaries_senior_analyst_20_words_gpt.csv",
        //         options.outputPathFor("SENIOR_ANALYST_20_WORDS"));
        // CHANGE: length variants are named like the normal run with the length appended
        assertEquals("output/summary-output/llm_summaries_gpt_20.csv", options.outputPathFor("SENIOR_ANALYST_20_WORDS"));
        assertEquals("output/summary-output/llm_summaries_mistral_80.csv",
                LlmRunOptions.parse(new String[] {"--mistral"}).outputPathFor("SENIOR_ANALYST_80_WORDS"));
        assertEquals("output/summary-output/llm_summaries_custom_alias_gpt.csv", options.outputPathFor("CUSTOM_ALIAS"));
    }

    @Test
    void resumeFlagKeepsTheSameOutputFile() {
        LlmRunOptions options = LlmRunOptions.parse(new String[] {"--mistral", "--nonconcise", "--resume"});
        assertTrue(options.resume());
        assertEquals("output/summary-output/llm_summaries_nonconcise_mistral.csv", options.defaultOutputPath());
        assertThrows(IllegalArgumentException.class, () -> LlmRunOptions.parse(new String[] {"--resume"}));
    }

    @Test
    void stillRequiresExactlyOneModel() {
        assertThrows(IllegalArgumentException.class, () -> LlmRunOptions.parse(new String[] {"--nonconcise"}));
        assertThrows(IllegalArgumentException.class, () -> LlmRunOptions.parse(new String[] {"--gpt", "--mistral"}));
        assertThrows(IllegalArgumentException.class, () -> LlmRunOptions.parse(new String[] {"--gpt", "--concise"}));
    }
}
