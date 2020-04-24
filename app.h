#ifndef VK_APP_H
#define VK_APP_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.hpp>

#include <optional>

struct QueueFamilies {
	std::optional<uint32_t> graphics_family;
	std::optional<uint32_t> present_family;

	bool is_complete() {
		return graphics_family.has_value() && present_family.has_value();
	}
};

struct SwapchainSupport {
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> present_modes;
};

class VkApp {

private:
	bool validation_enabled_;

	GLFWwindow* window_;

	vk::Instance instance_;
	vk::DebugUtilsMessengerEXT dbg_messenger_;
	vk::SurfaceKHR surface_;

	vk::PhysicalDevice physical_device_;
	QueueFamilies queue_family_indices_;
	vk::Device device_;
	vk::Queue graphics_queue_;
	vk::Queue present_queue_;

	vk::SurfaceFormatKHR swap_format_;
	vk::PresentModeKHR swap_present_mode_;
	vk::Extent2D swap_extent_;
	vk::SwapchainKHR swapchain_;
	std::vector<vk::Image> swapchain_images_;
	std::vector<vk::ImageView> swapchain_image_views_;

	vk::RenderPass render_pass_;
	vk::PipelineLayout pipeline_layout_;
	vk::Pipeline graphics_pipeline_;

	std::vector<vk::Framebuffer> swap_framebuffers_;

	vk::CommandPool cmd_pool_;
	std::vector<vk::CommandBuffer> cmd_buffers_;

	std::vector<vk::Semaphore> img_available_semaphores_;
	std::vector<vk::Semaphore> render_finished_semaphores_;
	std::vector<vk::Fence> in_flight_fences_;
	std::vector<vk::Fence> imgs_in_flight_;
	size_t current_frame_ = 0;

	bool framebuffer_resized = false;

public:
	VkApp(int width, int height, std::string title, bool validation_enabled=false);
	~VkApp();

	void run();

	static VKAPI_ATTR VkBool32 VKAPI_CALL dbg_callback(
		VkDebugUtilsMessageSeverityFlagBitsEXT severity,
		VkDebugUtilsMessageTypeFlagsEXT type,
		const VkDebugUtilsMessengerCallbackDataEXT* cb_data,
		void* user_data
	);

	static void framebuffer_resize_callback(GLFWwindow* window, int width, int height);

private:

	void init_vulkan();

	void init_instance();
	void init_dbg_messenger();
	void init_surface();
	void select_physical_device();
	void init_logical_device();
	void init_swapchain();
	void init_image_views();
	void init_render_pass();
	void init_pipeline();
	void init_framebuffers();
	void init_cmd_pool();
	void init_cmd_buffers();
	void init_sync_objects();

	std::vector<const char*> get_required_exts() const noexcept;

	bool is_device_suitable(vk::PhysicalDevice device) const noexcept;
	QueueFamilies get_queue_families(vk::PhysicalDevice device) const noexcept;
	bool supports_extensions(vk::PhysicalDevice device) const noexcept;

	SwapchainSupport get_swapchain_support(vk::PhysicalDevice device) const noexcept;
	vk::SurfaceFormatKHR select_swap_surface_format(std::vector<vk::SurfaceFormatKHR> formats) const noexcept;
	vk::PresentModeKHR select_swap_present_mode(std::vector<vk::PresentModeKHR> modes) const noexcept;
	vk::Extent2D select_swap_extent(vk::SurfaceCapabilitiesKHR capabilities) const noexcept;

	vk::ShaderModule create_shader_module(std::vector<char> code);

	void draw_frame();

	void cleanup_swapchain();
	void recreate_swapchain();
};

#endif