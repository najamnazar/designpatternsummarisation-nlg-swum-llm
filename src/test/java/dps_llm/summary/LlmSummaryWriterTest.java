package dps_llm.summary;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Tests the --resume mode of the LLM summary CSV writer. */
class LlmSummaryWriterTest {

    @TempDir
    Path dir;

    @Test
    void resumeKeepsExistingRowsAndAppends() throws IOException {
        Path csv = dir.resolve("out.csv");
        Files.writeString(csv, "Project,Folder Name,File Name,Summary\n"
                + "\"ProjA\",\"Singleton\",\"A.java\",\"Says \"\"hi\"\", then exits.\"\n", StandardCharsets.UTF_8);

        try (LlmSummaryWriter writer = new LlmSummaryWriter(csv.toString(), true)) {
            assertEquals(1, writer.existingRowCount());
            assertTrue(writer.hasRow("ProjA/Singleton", "A.java"));
            assertFalse(writer.hasRow("ProjA/Singleton", "B.java"));
            writer.writeRow("ProjA/Singleton", "B.java", "New summary.");
        }

        List<String> lines = Files.readAllLines(csv, StandardCharsets.UTF_8);
        assertEquals(3, lines.size());
        assertEquals("\"ProjA\",\"Singleton\",\"B.java\",\"New summary.\"", lines.get(2));
    }

    @Test
    void resumeAddsMissingNewlineBeforeAppending() throws IOException {
        Path csv = dir.resolve("out.csv");
        Files.writeString(csv, "Project,Folder Name,File Name,Summary\n\"P\",\"F\",\"A.java\",\"x\"", StandardCharsets.UTF_8);
        try (LlmSummaryWriter writer = new LlmSummaryWriter(csv.toString(), true)) {
            writer.writeRow("P/F", "B.java", "y");
        }
        assertEquals(3, Files.readAllLines(csv, StandardCharsets.UTF_8).size());
    }

    @Test
    void withoutResumeTheCsvStartsFresh() throws IOException {
        Path csv = dir.resolve("out.csv");
        Files.writeString(csv, "Project,Folder Name,File Name,Summary\n\"P\",\"F\",\"A.java\",\"x\"\n", StandardCharsets.UTF_8);
        try (LlmSummaryWriter writer = new LlmSummaryWriter(csv.toString(), false)) {
            assertFalse(writer.hasRow("P/F", "A.java"));
        }
        assertEquals(List.of("Project,Folder Name,File Name,Summary"), Files.readAllLines(csv, StandardCharsets.UTF_8));
    }

    @Test
    void resumeOnMissingFileCreatesItWithHeader() throws IOException {
        Path csv = dir.resolve("sub/new.csv");
        try (LlmSummaryWriter writer = new LlmSummaryWriter(csv.toString(), true)) {
            assertEquals(0, writer.existingRowCount());
        }
        assertEquals(List.of("Project,Folder Name,File Name,Summary"), Files.readAllLines(csv, StandardCharsets.UTF_8));
    }
}
