import trimesh
import os
from trimesh.viewer.windowed import SceneViewer
import numpy as np
import pyglet
import shutil
from PIL import Image
from rearrangement_utils import make_gifs
import time

class CustomViewer(SceneViewer):

    pass

    # def set_scene(self, pcr, discriminator_inference=None, save_dir=None):
    #     # initialize some additional parameters for the custom viewer
    #     self.pcr = pcr
    #     self.obj_to_move = None
    #     self.discriminator_inference = discriminator_inference
    #     self.img_count = 0
    #     self.save_dir = save_dir
    #     self.discriminator_scores = []
    #
    #     if self.save_dir:
    #         if not os.path.exists(self.save_dir):
    #             os.makedirs(self.save_dir)
    #
    #     self.update_scene_with_pcr()
    #     time.sleep(1.0)
    #     self.save_current_scene_to_img(self.score_current_scene())
    #
    #     print("\n" + "*"*50)
    #     print("Starting custom trimesh visualizer...")
    #     print("press number key (e.g., 1, 2, 3) to select object")
    #     print("press y or u to change x")
    #     print("press h or j to change y")
    #     print("press n or m to change z")
    #     print("press i or o to change r")
    #     print("press k or l to change p")
    #     print("press , or . to change y")
    #     print("press s to save current scene to image buffer")
    #     print("press d to make a gif from buffered images")
    #     print("*" * 50 + "\n")
    #
    # def save_current_scene_to_img(self, score=None):
    #     if self.save_dir:
    #         print("saving current scene")
    #         self.save_image(os.path.join(self.save_dir, "{}.png").format(self.img_count))
    #         self.img_count += 1
    #         if score is not None:
    #             self.discriminator_scores.append(score)
    #
    # def make_gif(self):
    #     if self.save_dir:
    #         img_names = []
    #         for file in os.listdir(self.save_dir):
    #             if ".png" in file:
    #                 img_names.append(file)
    #         if img_names:
    #             print("Saving a gif to", self.save_dir)
    #             img_names = sorted(img_names, key=lambda x: int(x.split(".")[0]))
    #             imgs = [Image.open(os.path.join(self.save_dir, img_name)) for img_name in img_names]
    #
    #             if self.discriminator_scores:
    #                 texts = ["scene {} score {}".format(i + 1, self.discriminator_scores[i]) for i in range(len(img_names))]
    #             else:
    #                 texts = [str(i + 1) for i in range(len(img_names))]
    #
    #             make_gifs(imgs, os.path.join(self.save_dir, "arrangements.gif"), texts=texts,
    #                       numpy_img=False, duration=50)
    #             for img_name in img_names:
    #                 os.remove(os.path.join(self.save_dir, img_name))
    #         else:
    #             print("No buffer saved to make a gif.")
    #
    # def score_current_scene(self, multiple_times=20):
    #     score = None
    #     if self.discriminator_inference:
    #         if multiple_times > 0:
    #             beam_xyzs = []
    #             for i in range(multiple_times):
    #                 current_xyzs = {"xyzs": self.pcr.goal_xyzs["xyzs"]}
    #                 if self.pcr.initial_scene and self.pcr.obj_ids:
    #                     current_xyzs["initial_scene"] = self.pcr.initial_scene
    #                     current_xyzs["obj_ids"] = self.pcr.obj_ids
    #                 if self.pcr.current_obj_idx is not None:
    #                     current_xyzs["current_obj_idx"] = self.pcr.current_obj_idx
    #                 if self.pcr.sentence is not None and self.pcr.sentence_pad_mask is not None:
    #                     current_xyzs["sentence"] = self.pcr.sentence
    #                     current_xyzs["sentence_pad_mask"] = self.pcr.sentence_pad_mask
    #                 beam_xyzs.append(current_xyzs)
    #
    #             num_target_objects = self.pcr.num_target_objects
    #             scores = self.discriminator_inference.limited_batch_inference(beam_xyzs, num_target_objects)
    #             print("scores:", scores)
    #             score = np.mean(scores)
    #             print("score:", score)
    #         else:
    #             beam_xyzs = []
    #             current_xyzs = {"xyzs": self.pcr.goal_xyzs["xyzs"]}
    #             if self.pcr.initial_scene and self.pcr.obj_ids:
    #                 current_xyzs["initial_scene"] = self.pcr.initial_scene
    #                 current_xyzs["obj_ids"] = self.pcr.obj_ids
    #             if self.pcr.current_obj_idx is not None:
    #                 current_xyzs["current_obj_idx"] = self.pcr.current_obj_idx
    #             if self.pcr.sentence is not None and self.pcr.sentence_pad_mask is not None:
    #                 current_xyzs["sentence"] = self.pcr.sentence
    #                 current_xyzs["sentence_pad_mask"] = self.pcr.sentence_pad_mask
    #             beam_xyzs.append(current_xyzs)
    #             num_target_objects = self.pcr.num_target_objects
    #             score = self.discriminator_inference.limited_batch_inference(beam_xyzs, num_target_objects)[0]
    #             print("score:", score)
    #     return score
    #
    # def update_scene_with_pcr(self):
    #     xyzs = self.pcr.goal_xyzs["xyzs"]
    #     rgbs = self.pcr.goal_xyzs["rgbs"]
    #
    #     for i in range(len(xyzs)):
    #         geom_name = "object_pc_{}".format(i)
    #         if geom_name in self.scene.geometry:
    #             self.scene.delete_geometry("object_pc_{}".format(i))
    #
    #     for i, (obj_xyz, obj_rgb) in enumerate(zip(xyzs, rgbs)):
    #         obj_rgb_new = np.asarray(obj_rgb)
    #         obj_rgb_new = np.hstack([obj_rgb_new, np.ones([obj_rgb_new.shape[0], 1])])
    #         pc = trimesh.PointCloud(vertices=obj_xyz, colors=obj_rgb_new)
    #         self.scene.add_geometry(pc, geom_name="object_pc_{}".format(i))
    #
    # def on_key_press(self, symbol, modifiers):
    #     # call the parent method in SceneViewer
    #     super(self.__class__, self).on_key_press(
    #         symbol, modifiers)
    #     # print(symbol, modifiers)
    #     # print(self.scene.geometry.items())
    #     # print("pcr", self.pcr)
    #
    #     num_target_objects = self.pcr.num_target_objects
    #     goal_struct_pose, goal_obj_poses = self.pcr.get_goal_poses(output_pose_format="flat:xyz+rpy")
    #     goal_obj_poses = np.array(goal_obj_poses).reshape(num_target_objects, -1).tolist()
    #
    #     # print("goal struct pose", goal_struct_pose)
    #     # for i in range(num_target_objects):
    #     #     print("obj {} pose".format(i), goal_obj_poses[i])
    #
    #     if symbol == pyglet.window.key.S:
    #         self.save_current_scene_to_img(self.score_current_scene())
    #     if symbol == pyglet.window.key.D or symbol == pyglet.window.key.Q:
    #         self.make_gif()
    #
    #     for i in range(10):
    #         if symbol == i + 48:
    #             self.obj_to_move = i
    #             print("set to move obj {}".format(i))
    #
    #     if self.obj_to_move is not None and self.obj_to_move < len(goal_obj_poses):
    #         direction = 1.0
    #         dimension = None
    #         if symbol == pyglet.window.key.Y:
    #             dimension = 0
    #         if symbol == pyglet.window.key.U:
    #             dimension = 0
    #             direction = -1.0
    #         if symbol == pyglet.window.key.H:
    #             dimension = 1
    #         if symbol == pyglet.window.key.J:
    #             dimension = 1
    #             direction = -1.0
    #         if symbol == pyglet.window.key.N:
    #             dimension = 2
    #         if symbol == pyglet.window.key.M:
    #             dimension = 2
    #             direction = -1.0
    #         if symbol == pyglet.window.key.I:
    #             dimension = 3
    #         if symbol == pyglet.window.key.O:
    #             dimension = 3
    #             direction = -1.0
    #         if symbol == pyglet.window.key.K:
    #             dimension = 4
    #         if symbol == pyglet.window.key.L:
    #             dimension = 4
    #             direction = -1.0
    #         if symbol == 44:  # <
    #             dimension = 5
    #         if symbol == 46:  # >
    #             dimension = 5
    #             direction = -1.0
    #
    #         if dimension is not None:
    #             # print("before", goal_obj_poses[0])
    #             if dimension < 3:
    #                 goal_obj_poses[self.obj_to_move][dimension] += 0.01 * direction
    #             else:
    #                 goal_obj_poses[self.obj_to_move][dimension] += (np.pi / 180 * 5) * direction
    #             # print("after", goal_obj_poses[0])
    #
    #             goal_obj_poses = np.array(goal_obj_poses).flatten().tolist()
    #             self.pcr.set_goal_poses(goal_struct_pose, goal_obj_poses,
    #                                input_pose_format="flat:xyz+rpy")
    #             self.pcr.rearrange()
    #             self.update_scene_with_pcr()
    #             self.score_current_scene()

    # label = pyglet.text.Label('Hello, world',
    #                           font_name='Times New Roman',
    #                           font_size=100,
    #                           color=(255, 0, 0, 255),
    #                           x=self.width // 2, y=self.height // 2,
    #                           anchor_x='center', anchor_y='center')
    # label.draw()


def custom_trimesh_visualize(pcr, discriminator_inference=None, save_dir=None):
    scene = trimesh.Scene()
    # add the coordinate frame first
    geom = trimesh.creation.axis(0.01)
    scene.add_geometry(geom)
    table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
    table.apply_translation([0.5, 0, -0.02])
    scene.add_geometry(table)

    viewer = CustomViewer(scene, start_loop=False)
    viewer.set_scene(pcr, discriminator_inference, save_dir)
    def callback(dt, viewer):
        viewer._update_vertex_list()
        viewer.on_resize(viewer.width, viewer.height)  # to adjust far
    pyglet.clock.schedule_interval(callback, 1. / 20, viewer)
    pyglet.app.run()


# window = pyglet.window.Window()
# label = pyglet.text.Label('Hello, world',
#                           font_name='Times New Roman',
#                           font_size=36,
#                           x=window.width//2, y=window.height//2,
#                           anchor_x='center', anchor_y='center')
#
# @window.event
# def on_draw():
#     window.clear()
#     label.draw()
#
# pyglet.app.run()


def custom_trimesh_visualize():
    scene = trimesh.Scene()
    # add the coordinate frame first
    geom = trimesh.creation.axis(0.01)
    scene.add_geometry(geom)
    # table = trimesh.creation.box(extents=[1.0, 1.0, 0.02])
    # table.apply_translation([0.5, 0, -0.02])
    # scene.add_geometry(table)

    window = CustomViewer(scene, start_loop=False)
    window = SceneViewer(scene)
    label = pyglet.text.Label('Hello, world',
                              font_name='Times New Roman',
                              font_size=36,
                              x=window.width // 2, y=window.height // 2,
                              anchor_x='left', anchor_y='top')

    @window.event
    def on_draw():
        window.clear()
        label.draw()

    pyglet.app.run()


if __name__ == "__main__":
    custom_trimesh_visualize()