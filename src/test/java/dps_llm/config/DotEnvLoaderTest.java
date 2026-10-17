package dps_llm.config;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Tests that .env parsing matches python-dotenv for the cases this project uses. */
class DotEnvLoaderTest {

    @TempDir
    Path dir;

    @Test
    void parsesPlainQuotedAndExportedValues() throws IOException {
        Path env = dir.resolve(".env");
        Files.writeString(env, String.join("\n",
                "# comment",
                "",
                "PLAIN=abc",
                "DOUBLE=\"sk-123\"",
                "SINGLE='x y'",
                "export EXPORTED=1",
                "EMPTY=",
                "URL=https://example.org/a?b=c"));

        Map<String, String> values = DotEnvLoader.load(env);

        assertEquals("abc", values.get("PLAIN"));
        assertEquals("sk-123", values.get("DOUBLE"));
        assertEquals("x y", values.get("SINGLE"));
        assertEquals("1", values.get("EXPORTED"));
        assertEquals("", values.get("EMPTY"));
        assertEquals("https://example.org/a?b=c", values.get("URL"));
    }

    @Test
    void mismatchedQuotesAreKept() {
        assertEquals("\"abc'", DotEnvLoader.stripMatchingQuotes("\"abc'"));
        assertEquals("\"", DotEnvLoader.stripMatchingQuotes("\""));
    }

    @Test
    void missingFileGivesEmptyMap() throws IOException {
        assertTrue(DotEnvLoader.load(dir.resolve("absent.env")).isEmpty());
    }
}
