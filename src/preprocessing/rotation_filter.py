import torch
import torch.nn as nn
import torch.nn.functional as F

class RotateConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, symmetries=[2, 3, 4, 6], padding=1, stride=1, keep_original=False, trainable=True):
        super(RotateConv2d, self).__init__()
        self.in_channels = in_channels

        if keep_original == False:
            if out_channels % len(symmetries) != 0:
                raise ValueError("The number of output channels must be a multiple of the number of symmetries.")
            self.out_channels = int(out_channels//len(symmetries))
        elif keep_original:   
            if out_channels % (len(symmetries)*2) != 0:
                raise ValueError("The number of output channels must be a multiple of the number of symmetries.")
            self.out_channels = int(out_channels//(len(symmetries)*2))

        self.kernel_size = kernel_size
        self.symmetries = symmetries
        self.padding = padding
        self.stride = stride
        self.keep_original = keep_original

        # Initialize the weight for the original kernel
        self.weight = nn.Parameter(torch.randn(self.out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(self.out_channels))

        if not trainable:
            self.weight.requires_grad = False
            self.bias.requires_grad = False

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels * len(self.symmetries)}, kernel_size={self.kernel_size}, symmetries={self.symmetries}, padding={self.padding}, stride={self.stride})"

    def rotate_kernel(self, kernel, angle):
        # Convert angle from degrees to radians
        angle_rad = torch.tensor(angle * (torch.pi / 180)).to(kernel.device)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        # Define rotation matrix for 2D (ignoring translation part for simplicity)
        rotation_matrix = torch.tensor([[cos_a, sin_a], 
                                        [-sin_a, cos_a]], 
                                        device=kernel.device)
        # Expand rotation matrix to match batch size and add batch dimension and extra zeros for affine_grid
        rotation_matrix = rotation_matrix.unsqueeze(0).repeat(kernel.size(0), 1, 1)
        theta = torch.cat((rotation_matrix, torch.zeros(kernel.size(0), 2, 1, device=kernel.device)), dim=2)

        # Create an affine grid and sample the rotated kernel
        flow_field = F.affine_grid(theta, kernel.size(), align_corners=False)
        rotated_kernel = F.grid_sample(kernel, flow_field, mode='bilinear', padding_mode='zeros', align_corners=False)

        return rotated_kernel


    def forward(self, x):

        outputs_all = []
        # Apply convolutional layers for each type of symmetry
        for symmetry in self.symmetries:

            outputs = []
            num_rotations = symmetry
            for i in range(num_rotations):
                angle = 360 / num_rotations * i
                rotated_weight = self.rotate_kernel(self.weight, angle)
                output = F.conv2d(x, rotated_weight, self.bias, self.stride, self.padding)
                outputs.append(output)
            # Combine the outputs by taking 1 minus the standard deviation
            outputs = torch.stack(outputs)  # Shape: [num_rotations, batch_size, self.out_channels, new_height, new_width]
            std_dev = torch.std(outputs, dim=0)
            std_ = 1 - std_dev
            if self.keep_original:
                final_output = torch.cat([std_, output], dim=1)
            else:
                final_output = std_
            # print(final_output.shape)
            outputs_all.append(final_output)

        # Combine the outputs (e.g., by taking the maximum across symmetry operations)
        combined_output = torch.cat(outputs_all, dim=1)

        return combined_output


# class RotateConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, num_rotations=4, padding=1, stride=1):
#         super(RotateConv2d, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.num_rotations = num_rotations
#         self.padding = padding
#         self.stride = stride

#         # Initialize the weight for the original kernel
#         self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
#         self.bias = nn.Parameter(torch.zeros(out_channels))

#     def rotate_kernel(self, kernel, angle):
#         # Convert angle from degrees to radians
#         angle_rad = torch.tensor(angle * (torch.pi / 180)).to(kernel.device)
#         cos_a = torch.cos(angle_rad)
#         sin_a = torch.sin(angle_rad)

#         # Define rotation matrix for 2D (ignoring translation part for simplicity)
#         rotation_matrix = torch.tensor([[cos_a, sin_a], 
#                                         [-sin_a, cos_a]], 
#                                         device=kernel.device)
#         # print(rotation_matrix)
#         # Expand rotation matrix to match batch size and add batch dimension and extra zeros for affine_grid
#         rotation_matrix = rotation_matrix.unsqueeze(0).repeat(kernel.size(0), 1, 1)
#         theta = torch.cat((rotation_matrix, torch.zeros(kernel.size(0), 2, 1, device=kernel.device)), dim=2)
#         # print(rotation_matrix.shape, theta.shape)
#         # print(theta)

#         # Create an affine grid and sample the rotated kernel
#         flow_field = F.affine_grid(theta, kernel.size(), align_corners=False)
#         rotated_kernel = F.grid_sample(kernel, flow_field, mode='bilinear', padding_mode='zeros', align_corners=False)

#         return rotated_kernel

#     def forward(self, x):
#         batch_size, channels, height, width = x.size()
#         outputs = []

#         # Generate and apply rotated kernels
#         for i in range(self.num_rotations):
#             angle = 360 / self.num_rotations * i
#             rotated_weight = self.rotate_kernel(self.weight, angle)
#             output = F.conv2d(x, rotated_weight, self.bias, self.stride, self.padding)
#             outputs.append(output)

#         # Combine the outputs by taking 1 minus the standard deviation
#         outputs = torch.stack(outputs)  # Shape: [num_rotations, batch_size, out_channels, new_height, new_width]
#         print(outputs.shape)
#         std_dev = torch.std(outputs, dim=0)
#         print(std_dev.shape)
#         final_output = 1 - std_dev
#         print(final_output.shape)

#         return final_output

if __name__ == "__main__":
    # Adjust the dimensions of weight to [batch_size, channels, height, width] for affine_grid compatibility
    # Example usage
    conv = RotateConv2d(in_channels=1, out_channels=1, kernel_size=3)
    input = torch.randn(2, 1, 28, 28)  # Example input tensor
    output = conv(input)
    print(output.shape)

    # Test case
    conv.weight.data = torch.tensor([[[[0., 1., 0.],
                                    [0., 1., 0.],
                                    [0., 1., 0.]]]])

    # Rotate the kernel by 90 degrees
    rotated_kernel = conv.rotate_kernel(conv.weight, 90)

    # Print the original and rotated kernels
    print("Original kernel:\n", conv.weight)
    print("Rotated kernel:\n", rotated_kernel)
    print("The expected result of the rotated kernel should be:\n", torch.tensor([[0., 0., 0.],
                                                                                 [1., 1., 1.],
                                                                                 [0., 0., 0.]]))
    # The expected result of the rotated kernel should be:
    # tensor([[[[0., 0., 0.],
    #            [1., 1., 1.],
    #            [0., 0., 0.]]]])