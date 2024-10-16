
from manim import *

# Existing Scene classes and functions

# Insert generated Scene classes below


# Generated Scene Classes
class SpaceIsALatentSequenceATheoryOfTheHippocampus(ThreeDScene):
    def construct(self):
        self.setup_scene()
        self.add_title()
        self.present_concept()
        self.show_examples()
        self.conclude_scene()

    def setup_scene(self):
        self.camera.background_color = DARK_GRAY
        self.axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(self.axes))

    def add_title(self):
        title = Text("Space is a Latent Sequence", font_size=72)
        subtitle = Text("A Theory of the Hippocampus", font_size=48)
        title.to_edge(UP)
        subtitle.next_to(title, DOWN)
        self.play(Write(title))
        self.play(Write(subtitle))
        self.wait(2)

    def present_concept(self):
        surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                np.sin(u) * np.cos(v)
            ]),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            resolution=(30, 30),
            fill_opacity=0.8,
            checkerboard_colors=[{BLUE_D}, {BLUE_E}],
        )
        self.play(Create(surface), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        concept_text = Text("Hippocampal Representation as a Sequence", font_size=36)
        concept_text.move_to([0, 0, 2])
        self.play(Write(concept_text))
        self.wait(2)

    def show_examples(self):
        vector = Vector([2, 1, 0], color=YELLOW)
        self.play(GrowArrow(vector))
        matrix = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        transformed_vector = vector.copy().apply_matrix(matrix)
        self.play(Transform(vector, transformed_vector))
        self.wait()

        graph = self.create_causal_graph()
        self.play(Create(graph))
        self.wait(2)

    def create_causal_graph(self):
        # Create nodes and edges for the CSCG model
        nodes = [
            Dot(point=np.array([x, y, 0]), color=WHITE)
            for x, y in product(range(-2, 3), repeat=2)
        ]
        edges = [
            Line(nodes[i].get_center(), nodes[j].get_center(), color=BLUE)
            for i in range(len(nodes)) for j in range(len(nodes)) if i != j
        ]
        return VGroup(*edges, *nodes)

    def conclude_scene(self):
        conclusion = Text("Conclusion: Space as a Sequence", font_size=64)
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(3)

class LearningAndInferenceInClonedStateConditionalGraphsCscg(ThreeDScene):
    def construct(self):
        self.setup_scene()
        self.add_title()
        self.present_concept()
        self.show_examples()
        self.conclude_scene()

    def setup_scene(self):
        self.camera.background_color = DARK_GRAY
        self.axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(self.axes))

    def add_title(self):
        title = Text("Learning and Inference in Cloned State Conditional Graphs", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.title = title

    def present_concept(self):
        concept_text = Text("CSCG Model for Sequence Learning", font_size=36, color=WHITE)
        concept_text.to_edge(UP)
        self.play(Write(concept_text))
        
        cloning_structure = Cube(side_length=1, fill_opacity=0.5, color=BLUE)
        self.play(Create(cloning_structure))
        self.wait(1)

        self.play(Rotate(cloning_structure, angle=PI/4, axis=UP), run_time=2)
        self.wait(1)

        latent_space = Surface(
            lambda u, v: np.array([u, v, np.sin(u) * np.cos(v)]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(30, 30),
            fill_opacity=0.5,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        self.play(Create(latent_space), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(3)
        self.stop_ambient_camera_rotation()

    def show_examples(self):
        action_sequence = Line(start=[-3, 0, 0], end=[3, 0, 0], color=YELLOW)
        self.play(Create(action_sequence))
        
        dynamic_environment = ParametricSurface(
            lambda u, v: np.array([u, v, np.sin(u) * np.cos(v)]),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            checkerboard_colors=[GREEN_D, GREEN_E],
            fill_opacity=0.5
        )
        self.play(Create(dynamic_environment), run_time=3)
        self.wait(2)

        self.play(FadeOut(action_sequence), FadeOut(dynamic_environment))
        self.wait(1)

    def conclude_scene(self):
        conclusion = Text("CSCGs: Effective in Learning Complex Maps", font_size=36, color=WHITE)
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)

class CscgsAndTheirRoleInNavigationAndPlaceFieldRemapping(ThreeDScene):
    def construct(self):
        self.setup_scene()
        self.add_title()
        self.present_concept()
        self.show_examples()
        self.conclude_scene()

    def setup_scene(self):
        self.camera.background_color = DARK_GRAY
        self.axes = ThreeDAxes()
        self.set_camera_orientation(phi=80 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(self.axes))

    def add_title(self):
        title = Text("CSCGs and Their Role in Navigation", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.title = title

    def present_concept(self):
        graph = Surface(
            lambda u, v: np.array([
                u,
                v,
                np.sin(u) * np.cos(v)
            ]),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(30, 30),
            fill_opacity=0.6,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        self.play(Create(graph), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(5)
        self.stop_ambient_camera_rotation()

    def show_examples(self):
        transition_graph = VGroup(*[
            Line(start=[-2, -2, 0], end=[2, 2, 0], color=YELLOW),
            Line(start=[-2, 2, 0], end=[2, -2, 0], color=YELLOW)
        ])
        self.play(Create(transition_graph))
        self.wait(2)
        
        emission_matrix = Matrix([[0.1, 0.9], [0.8, 0.2]], element_alignment='center')
        emission_matrix.move_to([3, 0, 0])
        self.play(Create(emission_matrix))
        self.wait(2)

        self.play(FadeOut(transition_graph), FadeOut(emission_matrix))

    def conclude_scene(self):
        conclusion = Text("CSCGs: Adapting to Spatial Changes", font_size=64)
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(3)

class SequenceContextsAndPlaceFieldRemappingInSpatialNavigation(ThreeDScene):
    def construct(self):
        self.setup_scene()
        self.add_title()
        self.present_concept()
        self.show_examples()
        self.conclude_scene()

    def setup_scene(self):
        self.camera.background_color = DARK_GRAY
        self.axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(self.axes))

    def add_title(self):
        title = Text("Sequence Contexts and Place Field Remapping", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.title = title

    def present_concept(self):
        concept_text = Text("Exploring place field remapping in spatial navigation.", font_size=36)
        concept_text.move_to(ORIGIN)
        self.play(Write(concept_text))
        
        # Visualizing place fields
        place_fields = VGroup(*[
            Circle(radius=0.5, color=BLUE).shift(np.array([np.cos(angle), np.sin(angle), 0]))
            for angle in np.linspace(0, 2 * PI, 8)
        ])
        self.play(Create(place_fields))
        self.wait(2)
        
        # Animate remapping
        for field in place_fields:
            field.generate_target()
            field.target.shift(UP * 2)
        self.play(MoveToTarget(place_fields), run_time=2)
        self.wait(1)

    def show_examples(self):
        # Show direction sensitivity
        direction_vector = Vector([1, 0, 0], color=YELLOW)
        self.play(GrowArrow(direction_vector))
        self.wait(1)

        # Visualize event-specific rate remapping
        remapping_surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                np.sin(u) * np.cos(v) + np.cos(u * v)
            ]),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            resolution=(30, 30),
            fill_opacity=0.8,
            checkerboard_colors=[{BLUE_D}, {BLUE_E}],
        )
        self.play(Create(remapping_surface), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        # Show place field variations
        variations = VGroup(*[
            Circle(radius=0.5 + 0.2 * np.random.rand(), color=GREEN).shift(np.array([np.cos(angle), np.sin(angle), 0]))
            for angle in np.linspace(0, 2 * PI, 8)
        ])
        self.play(Create(variations))
        self.wait(1)

    def conclude_scene(self):
        conclusion = Text("Conclusion: Context and connectivity shape navigation.", font_size=64)
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))

class CscgModelASequenceCentricApproachToUnderstandingPlaceCells(ThreeDScene):
    def construct(self):
        self.setup_scene()
        self.add_title()
        self.present_concept()
        self.show_examples()
        self.conclude_scene()

    def setup_scene(self):
        # Initialize the scene with a dark background and 3D axes
        self.camera.background_color = DARK_GRAY
        self.axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(self.axes))

    def add_title(self):
        # Add the title and subtitle
        title = Text("CSCG Model: A Sequence-Centric Approach to Understanding Place Cells", font_size=36)
        title.to_edge(UP)
        self.play(Write(title))
        self.title = title

    def present_concept(self):
        # Visualize the concept of place cells and their sequential context
        sequence_surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                np.sin(u) * np.cos(v)  # Example of a dynamic surface
            ]),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            resolution=(30, 30),
            fill_opacity=0.8,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        self.play(Create(sequence_surface), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        # Add text to explain the concept
        explanation = Text("Interpreting Spatial Representation as Sequences", font_size=24)
        explanation.next_to(self.title, DOWN)
        self.play(Write(explanation))
        self.wait(2)
        self.play(FadeOut(explanation))

    def show_examples(self):
        # Show examples of neuronal responses to environmental changes
        place_cell_vector = Vector([2, 1, 0], color=YELLOW)
        self.play(GrowArrow(place_cell_vector))
        
        # Simulate remapping phenomena
        transformed_vector = place_cell_vector.copy().shift(UP * 2)
        self.play(Transform(place_cell_vector, transformed_vector))
        self.wait(2)

        # Visualize place cells with a grid
        grid = VGroup(*[Square(side_length=0.5, color=WHITE).move_to([x, y, 0])
                        for x, y in product(np.arange(-3, 4), np.arange(-3, 4))])
        self.play(Create(grid), run_time=2)
        self.wait(2)

        # Highlight specific place cells
        highlight = Circle(radius=0.5, color=RED).move_to([-1, -1, 0])
        self.play(GrowFromCenter(highlight))
        self.wait(2)

    def conclude_scene(self):
        # Conclude the scene with a summary
        conclusion = Text("Dynamic Learning of Spatial Representations", font_size=36)
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(3)

class ReevaluatingHippocampalPlaceFieldMapping(ThreeDScene):
    def construct(self):
        self.setup_scene()
        self.add_title()
        self.present_concept()
        self.show_examples()
        self.conclude_scene()

    def setup_scene(self):
        self.camera.background_color = DARK_GRAY
        self.axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(self.axes))

    def add_title(self):
        title = Text("Reevaluating Hippocampal Place Field Mapping", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.title = title

    def present_concept(self):
        surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                np.sin(u) * np.cos(v)
            ]),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            resolution=(30, 30),
            fill_opacity=0.8,
            checkerboard_colors=[{BLUE_D}, {BLUE_E}],
        )
        self.play(Create(surface), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()

        text = Text("Hippocampal phenomena as artifacts?", font_size=48)
        text.next_to(surface, UP)
        self.play(Write(text))

    def show_examples(self):
        vector = Vector([2, 1, 0], color=YELLOW)
        self.play(GrowArrow(vector))
        matrix = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        transformed_vector = vector.copy().apply_matrix(matrix)
        self.play(Transform(vector, transformed_vector))
        self.wait()

        scatter_points = VGroup(*[
            Dot(point=[x, y, 0], color=WHITE)
            for x, y in np.random.rand(10, 2) * 4 - 2
        ])
        self.play(FadeIn(scatter_points))
        self.wait(2)

    def conclude_scene(self):
        conclusion = Text("Place field mapping: A visualization tool", font_size=64)
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)

class LearningClonedStructuredContinuousGraphsCscg(ThreeDScene):
    def construct(self):
        self.setup_scene()
        self.add_title()
        self.present_concept()
        self.show_examples()
        self.conclude_scene()

    def setup_scene(self):
        self.camera.background_color = DARK_GRAY
        self.axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(self.axes))

    def add_title(self):
        title = Text("Learning Cloned Structured Continuous Graphs (CSCG)", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.title = title

    def present_concept(self):
        # Visualizing the Expectation-Maximization (EM) process
        em_surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                np.exp(-u**2 - v**2) * np.cos(2 * np.pi * (u + v))
            ]),
            u_range=[-2, 2],
            v_range=[-2, 2],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        self.play(Create(em_surface), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.1)
        self.wait(5)
        self.stop_ambient_camera_rotation()

    def show_examples(self):
        # Example: Demonstrating K-means clustering in 3D
        cluster_points = VGroup(*[
            Dot3D(point=3 * np.random.rand(3) - 1.5, color=WHITE) for _ in range(10)
        ])
        self.play(FadeIn(cluster_points))
        self.wait(1)

        # Simulating K-means centroids
        centroids = VGroup(
            Dot3D(point=[-1, -1, 0], color=YELLOW),
            Dot3D(point=[1, 1, 0], color=RED)
        )
        self.play(FadeIn(centroids))
        
        for _ in range(2):  # Simulate K-means steps
            self.play(ApplyMethod(cluster_points.move_to, lambda: np.random.uniform(-2, 2, (10, 3))), run_time=2)
            self.wait(1)

        self.play(FadeOut(cluster_points), FadeOut(centroids))

    def conclude_scene(self):
        conclusion = Text("Conclusion: Effective Learning with CSCG", font_size=64)
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))

class HippocampalPlaceCellsAndSpatialMemory(ThreeDScene):
    def construct(self):
        self.setup_scene()
        self.add_title()
        self.present_concept()
        self.show_examples()
        self.conclude_scene()

    def setup_scene(self):
        # Initialize the 3D axes and set the camera orientation
        self.camera.background_color = DARK_GRAY
        self.axes = ThreeDAxes()
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.play(Create(self.axes))

    def add_title(self):
        # Add title and subtitle
        title = Text("Hippocampal Place Cells and Spatial Memory", font_size=48, color=WHITE)
        title.to_edge(UP)
        self.play(Write(title))
        self.title = title

    def present_concept(self):
        # Visualize the concept of place cells in a 3D space
        surface = Surface(
            lambda u, v: np.array([
                u,
                v,
                np.sin(np.sqrt(u**2 + v**2))  # Simulating a spatial representation
            ]),
            u_range=[-3, 3],
            v_range=[-3, 3],
            resolution=(30, 30),
            fill_opacity=0.7,
            checkerboard_colors=[BLUE_D, BLUE_E],
        )
        self.play(Create(surface), run_time=3)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)
        self.stop_ambient_camera_rotation()

    def show_examples(self):
        # Show examples of how place cells respond to environmental changes
        place_cell = Dot3D(point=[0, 0, 0], color=YELLOW, radius=0.1)
        self.play(FadeIn(place_cell))
        self.wait(1)

        # Simulating the movement of the place cell in a spatial environment
        path = Line3D(start=[0, 0, 0], end=[2, 2, 1])
        self.play(MoveAlongPath(place_cell, path), run_time=3)
        self.wait(1)

        # Show the influence of sensory cues
        cue = Circle(radius=0.3, color=RED).shift([2, 2, 0])
        self.play(Create(cue))
        self.wait(1)
        self.play(place_cell.animate.move_to(cue.get_center() + [0, 0, 1]), run_time=2)
        self.wait()

    def conclude_scene(self):
        # Conclude the scene
        conclusion = Text("Conclusion: Place Cells and Memory", font_size=36, color=WHITE)
        conclusion.to_edge(DOWN)
        self.play(Write(conclusion))
        self.wait(2)

