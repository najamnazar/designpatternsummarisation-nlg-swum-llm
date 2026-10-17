package common.utils;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;

import org.junit.jupiter.api.Test;

import com.github.javaparser.StaticJavaParser;
import com.github.javaparser.ast.body.MethodDeclaration;

/** Tests for Utils.getParameters, which used to split parameter text on spaces. */
class UtilsParametersTest {

    private static List<String> parse(String method) {
        MethodDeclaration declaration = StaticJavaParser.parseMethodDeclaration(method);
        ArrayList<HashMap> parameters = Utils.getParameters(declaration.getParameters());
        return parameters.stream()
                .map(p -> Utils.getParameterType(p) + "|" + Utils.getParameterName(p))
                .collect(Collectors.toList());
    }

    @Test
    void simpleParameters() {
        assertEquals(List.of("String|name", "int|count"), parse("void m(String name, int count) {}"));
    }

    @Test
    void annotatedParameterKeepsRealTypeAndName() {
        // Old behaviour: type "@Nullable", name "BeanFactory"
        assertEquals(List.of("BeanFactory|beanFactory"), parse("void m(@Nullable BeanFactory beanFactory) {}"));
    }

    @Test
    void finalModifierIsNotTheType() {
        // Old behaviour: type "final", name "String"
        assertEquals(List.of("String|value"), parse("void m(final String value) {}"));
    }

    @Test
    void genericTypeWithSpacesStaysWhole() {
        // Old behaviour: type "Map<String,", name "Object>"
        assertEquals(List.of("Map<String, Object>|attributes"), parse("void m(Map<String, Object> attributes) {}"));
    }

    @Test
    void varargsKeepEllipsisLikeBefore() {
        assertEquals(List.of("String...|args"), parse("void m(String... args) {}"));
    }
}
