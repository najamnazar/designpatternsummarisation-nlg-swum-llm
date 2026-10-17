package common.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/** Tests for how project identifiers become the Project and Folder Name CSV columns. */
class ProjectPathFormatterTest {

    @Test
    void slashSeparatedPath() {
        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split("AbdurRKhalid/Abstract Factory");
        assertEquals("AbdurRKhalid", parts.projectName());
        assertEquals("Abstract Factory", parts.folderName());
    }

    @Test
    void underscoreFallbackWhenNoSlash() {
        // CHANGE (naming consistency): sample uses the SpringFramework / Adapter labels.
        // ProjectPathFormatter.Parts parts = ProjectPathFormatter.split("spring-framework_spring-aop");
        // assertEquals("spring-framework", parts.projectName());
        // assertEquals("spring-aop", parts.folderName());
        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split("SpringFramework_Adapter");
        assertEquals("SpringFramework", parts.projectName());
        assertEquals("Adapter", parts.folderName());
    }

    @Test
    void projectWithoutFolder() {
        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split("JamesZBL");
        assertEquals("JamesZBL", parts.projectName());
        assertEquals("", parts.folderName());
    }

    @Test
    void blankInput() {
        ProjectPathFormatter.Parts parts = ProjectPathFormatter.split("  ");
        assertEquals("", parts.projectName());
        assertEquals("", parts.folderName());
    }
}
