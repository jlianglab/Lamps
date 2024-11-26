   
import torch
import numpy as np
import torch.nn.functional as F
import utils
def _vicreg_loss(x, y, args):
    repr_loss = args.inv_coeff * F.mse_loss(x, y)

    x = utils.gather_center(x)
    y = utils.gather_center(y)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = args.var_coeff * (
        torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
    )

    x = x.permute((1, 0, 2))
    y = y.permute((1, 0, 2))

    *_, sample_size, num_channels = x.shape
    non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
    # Center features
    # centered.shape = NC
    x = x - x.mean(dim=-2, keepdim=True)
    y = y - y.mean(dim=-2, keepdim=True)

    cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
    cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
    cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
        cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
    ) / 2
    cov_loss = cov_loss.mean()
    cov_loss = args.cov_coeff * cov_loss

    return repr_loss, std_loss, cov_loss

def _local_loss(maps_1, maps_2, location_1, location_2,args):
    inv_loss = 0.0
    var_loss = 0.0
    cov_loss = 0.0

    #L2 distance based bacthing
    if args.l2_all_matches:
        num_matches_on_l2 = [None, None]
    else:
        num_matches_on_l2 = args.num_matches

    maps_1_filtered, maps_1_nn = utils.neirest_neighbores_on_l2(
        maps_1, maps_2, num_matches=num_matches_on_l2[0]
    )
    maps_2_filtered, maps_2_nn = utils.neirest_neighbores_on_l2(
        maps_2, maps_1, num_matches=num_matches_on_l2[1]
    )

    # if args.fast_vc_reg:
    inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
    inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
    # else:
    #     inv_loss_1, var_loss_1, cov_loss_1 = _vicreg_loss(maps_1_filtered, maps_1_nn, args)
    #     inv_loss_2, var_loss_2, cov_loss_2 = _vicreg_loss(maps_2_filtered, maps_2_nn, args)
    #     var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
    #     cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

    inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

    # Location based matching
    location_1 = location_1.flatten(1, 2)
    location_2 = location_2.flatten(1, 2)

    maps_1_filtered, maps_1_nn = utils.neirest_neighbores_on_location(
        location_1,
        location_2,
        maps_1,
        maps_2,
        num_matches=args.num_matches[0],
    )
    maps_2_filtered, maps_2_nn = utils.neirest_neighbores_on_location(
        location_2,
        location_1,
        maps_2,
        maps_1,
        num_matches=args.num_matches[1],
    )

    #if args.fast_vc_reg:
    inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
    inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
    # else:
    #     inv_loss_1, var_loss_1, cov_loss_1 = _vicreg_loss(maps_1_filtered, maps_1_nn, args)
    #     inv_loss_2, var_loss_2, cov_loss_2 = _vicreg_loss(maps_2_filtered, maps_2_nn, args)
    #     var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
    #     cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

    inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)

    return inv_loss#, var_loss, cov_loss

def local_loss(maps_embedding_student, maps_embedding_teacher, locations, args):
    num_views = len(maps_embedding_student)
    inv_loss = 0.0
    # var_loss = 0.0
    # cov_loss = 0.0
    iter_ = 0
    for i in range(2):
        for j in np.delete(np.arange(np.sum(num_views)), i):
            inv_loss_this = _local_loss(           ##, var_loss_this, cov_loss_this
                maps_embedding_teacher[i], maps_embedding_student[j], locations[i], locations[j],args
            )
            inv_loss = inv_loss + inv_loss_this
            # var_loss = var_loss + var_loss_this
            # cov_loss = cov_loss + cov_loss_this
            iter_ += 1

    # if args.fast_vc_reg:
    #     inv_loss = args.inv_coeff * inv_loss / iter_
    #     var_loss = 0.0
    #     cov_loss = 0.0
    #     iter_ = 0
    #     for i in range(num_views):
    #         x = utils.gather_center(maps_embedding[i])
    #         std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    #         var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
    #         x = x.permute(1, 0, 2)
    #         *_, sample_size, num_channels = x.shape
    #         non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
    #         x = x - x.mean(dim=-2, keepdim=True)
    #         cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
    #         cov_loss = cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels
    #         cov_loss = cov_loss + cov_loss.mean()
    #         iter_ = iter_ + 1
    #     var_loss = args.var_coeff * var_loss / iter_
    #     cov_loss = args.cov_coeff * cov_loss / iter_
    #else:
    inv_loss = inv_loss / iter_
    # var_loss = var_loss / iter_
    # cov_loss = cov_loss / iter_

    return inv_loss# , var_loss, cov_loss

def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)

def globalconsis_loss(student_output, teacher_output, grids):

    #print(output[0].shape,grids[0].shape)
    B,P,channels=student_output[0].shape
    inv_loss = 0
    #print(B,grids[0].shape[0])
    indices_selected = []
    for idx,grid in enumerate(grids):
        mask_thistime=grid
        mask_final=mask_thistime>=0.95
        mask_final=mask_final.reshape(mask_final.shape[0],-1)
        indices = (
            torch.arange(0, 49)
            .unsqueeze(0)
            .repeat(int(mask_final.shape[0]), 1)
            .to(student_output[0].device)
        )            
        indices_selected.append(indices.masked_select(mask_final))
    #print(indices_selected[0].shape)
    filtered_view1 = batched_index_select(student_output[0].reshape(-1,channels), 0, indices_selected[0])
    filtered_view2 = batched_index_select(teacher_output[1].reshape(-1,channels), 0, indices_selected[1])
    loss = F.mse_loss(filtered_view1, filtered_view2)
    inv_loss = inv_loss + loss


    indices_selected = []
    for idx,grid in enumerate(grids):
        mask_thistime=grid
        mask_final=mask_thistime>=0.95
        mask_final=mask_final.reshape(mask_final.shape[0],-1)
        indices = (
            torch.arange(0, 49)
            .unsqueeze(0)
            .repeat(int(mask_final.shape[0]), 1)
            .to(student_output[0].device)
        )            
        indices_selected.append(indices.masked_select(mask_final))
    #print(indices_selected[0].shape)
    filtered_view1 = batched_index_select(teacher_output[0].reshape(-1,channels), 0, indices_selected[0])
    filtered_view2 = batched_index_select(student_output[1].reshape(-1,channels), 0, indices_selected[1])

    loss = F.mse_loss(filtered_view1, filtered_view2)
    inv_loss = inv_loss + loss     




    return inv_loss 
