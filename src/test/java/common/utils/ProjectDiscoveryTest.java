package common.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Tests for deterministic project discovery and stale JSON cleanup. */
class ProjectDiscoveryTest {

    @TempDir
    Path root;

    @Test
    void projectDirectoriesAreReturnedInPathOrder() throws IOException {
        // Created in reverse order on purpose; the result must not depend on creation or filesystem order.
        for (String dir : new String[] {"P/FactoryMethod", "P/Factory Method", "P/AbstractFactory", "P/Abstract Factory"}) {
            Path path = Files.createDirectories(root.resolve(dir));
            Files.writeString(path.resolve("A.java"), "class A {}");
        }
        Files.createDirectories(root.resolve("P/empty"));

        List<String> names = ProjectDiscovery.findProjectDirectories(root.toFile()).stream()
                .map(File::getName)
                .collect(Collectors.toList());

        assertEquals(List.of("Abstract Factory", "AbstractFactory", "Factory Method", "FactoryMethod"), names);
    }

    @Test
    void deleteJsonFilesOnlyRemovesJson() throws IOException {
        Files.writeString(root.resolve("old.json"), "{}");
        Files.writeString(root.resolve("keep.txt"), "x");

        assertEquals(1, ProjectDiscovery.deleteJsonFiles(root.toFile()));
        assertEquals(List.of("keep.txt"), List.of(root.toFile().list()));
    }

    @Test
    void deleteJsonFilesIgnoresMissingDirectory() throws IOException {
        assertEquals(0, ProjectDiscovery.deleteJsonFiles(root.resolve("missing").toFile()));
    }
}
