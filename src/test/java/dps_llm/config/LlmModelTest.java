package dps_llm.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

/** Tests the --gpt / --mistral flag that selects the LLM pipeline's model. */
class LlmModelTest {

    @Test
    void acceptsDoubleAndSingleDashFlagsInAnyCase() {
        assertEquals(LlmModel.GPT, LlmModel.fromArgs(new String[] {"--gpt"}));
        assertEquals(LlmModel.GPT, LlmModel.fromArgs(new String[] {"-GPT"}));
        assertEquals(LlmModel.MISTRAL, LlmModel.fromArgs(new String[] {"--mistral"}));
        assertEquals(LlmModel.MISTRAL, LlmModel.fromArgs(new String[] {"-Mistral"}));
    }

    @Test
    void mapsEachModelToItsEnvKeyAndOwnOutputFile() {
        assertEquals("GPT_MODEL", LlmModel.GPT.envKey());
        assertEquals("MISTRAL_MODEL", LlmModel.MISTRAL.envKey());
        assertEquals("output/summary-output/llm_summaries_gpt.csv", LlmModel.GPT.defaultOutputPath());
        assertEquals("output/summary-output/llm_summaries_mistral.csv", LlmModel.MISTRAL.defaultOutputPath());
    }

    @Test
    void rejectsMissingConflictingOrUnknownFlags() {
        assertThrows(IllegalArgumentException.class, () -> LlmModel.fromArgs(new String[0]));
        assertThrows(IllegalArgumentException.class, () -> LlmModel.fromArgs(null));
        assertThrows(IllegalArgumentException.class, () -> LlmModel.fromArgs(new String[] {"--gpt", "--mistral"}));
        assertThrows(IllegalArgumentException.class, () -> LlmModel.fromArgs(new String[] {"--llama"}));
    }
}
