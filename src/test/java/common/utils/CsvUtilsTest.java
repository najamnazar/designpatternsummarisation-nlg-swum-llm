package common.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/** Tests for the shared CSV helpers that replaced the per-pipeline copies. */
class CsvUtilsTest {

    @Test
    void longSummariesAreNotTruncated() {
        String summary = "word ".repeat(400).trim(); // ~2000 characters, above the old 1000/600 caps
        assertEquals(summary, CsvUtils.escapeField(summary));
    }

    @Test
    void quotesAreDoubledAndNewlinesFlattened() {
        assertEquals("say \"\"hi\"\" now", CsvUtils.escapeField("  say \"hi\"\r\nnow  "));
    }

    @Test
    void nullBecomesEmptyField() {
        assertEquals("", CsvUtils.escapeField(null));
    }

    @Test
    void rowIsQuotedAndEndsWithUnixNewline() {
        assertEquals("\"A\",\"B C\",\"x.java\",\"a \"\"q\"\"\"\n",
                CsvUtils.formatRow("A", "B C", "x.java", "a \"q\""));
    }
}
